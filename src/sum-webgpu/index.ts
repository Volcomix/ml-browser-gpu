import './style.css'

if (!navigator.gpu) {
  throw Error('WebGPU not supported.')
}
const adapter = await navigator.gpu.requestAdapter()
if (!adapter) {
  throw Error("Couldn't request WebGPU adapter.")
}
const device = await adapter.requestDevice()

const generateInput = (count: number) => {
  const start = performance.now()
  const input = Int32Array.from({ length: count }, () =>
    Math.round(Math.random() * 10),
  )
  const time = performance.now() - start
  console.log(`${generateInput.name}: ${time} ms`)
  return input
}

type SumResult = {
  result: number
  time: number
}

const setupSumCPU = (input: Int32Array) => {
  const sumCPU = (): SumResult => {
    const start = performance.now()
    const result = input.reduce((a, b) => a + b)
    const time = performance.now() - start
    return { result, time }
  }

  return sumCPU
}

const setupSumReduction = async (input: Int32Array) => {
  const workgroupSize = Math.min(64, input.length)

  let workgroupCountX = input.length / workgroupSize
  let workgroupCountY = 1
  while (workgroupCountX > 65535) {
    workgroupCountX /= 2
    workgroupCountY *= 2
  }

  const clearModule = device.createShaderModule({
    code: /* wgsl */ `
      @group(0) @binding(0)
      var<storage, read_write> output: u32;

      @compute @workgroup_size(1)
      fn main() {
        output = 0u;
      }
    `,
  })

  const sumModule = device.createShaderModule({
    code: /* wgsl */ `
      @group(0) @binding(0)
      var<storage> input: array<u32>;

      @group(0) @binding(1)
      var<storage, read_write> output: atomic<u32>;

      var<workgroup> sharedData: array<u32, ${workgroupSize}>;

      @compute @workgroup_size(${workgroupSize})
      fn main(
        @builtin(workgroup_id)
        workgroupId: vec3u,

        @builtin(local_invocation_index)
        localIndex: u32,
      ) {
        let workgroupIndex = workgroupId.x + workgroupId.y * ${workgroupCountX}u;
        let i = workgroupIndex * ${workgroupSize}u + localIndex;
        sharedData[localIndex] = input[i];
        workgroupBarrier();

        for (var stride = ${workgroupSize}u / 2u; stride > 0u; stride >>= 1u) {
          if (localIndex < stride) {
            sharedData[localIndex] += sharedData[localIndex + stride];
          }
          workgroupBarrier();
        }

        if (localIndex == 0u) {
          atomicAdd(&output, sharedData[0]);
        }
      }
    `,
  })

  const inputBuffer = device.createBuffer({
    size: input.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(inputBuffer, 0, input)

  const outputBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  })
  const stagingBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  })

  const clearBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
    ],
  })
  const clearBindGroup = device.createBindGroup({
    layout: clearBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: outputBuffer } }],
  })
  const clearPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [clearBindGroupLayout],
  })
  const clearPipeline = device.createComputePipeline({
    layout: clearPipelineLayout,
    compute: { module: clearModule, entryPoint: 'main' },
  })

  const sumBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
    ],
  })
  const sumBindGroup = device.createBindGroup({
    layout: sumBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
    ],
  })
  const sumPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [sumBindGroupLayout],
  })
  const sumPipeline = device.createComputePipeline({
    layout: sumPipelineLayout,
    compute: { module: sumModule, entryPoint: 'main' },
  })

  const sumReduction = async (): Promise<SumResult> => {
    const start = performance.now()

    const encoder = device.createCommandEncoder()

    const clearPass = encoder.beginComputePass()
    clearPass.setPipeline(clearPipeline)
    clearPass.setBindGroup(0, clearBindGroup)
    clearPass.dispatchWorkgroups(1)
    clearPass.end()

    const sumPass = encoder.beginComputePass()
    sumPass.setPipeline(sumPipeline)
    sumPass.setBindGroup(0, sumBindGroup)
    sumPass.dispatchWorkgroups(workgroupCountX, workgroupCountY)
    sumPass.end()

    encoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, 4)

    const commands = encoder.finish()
    device.queue.submit([commands])

    await stagingBuffer.mapAsync(GPUMapMode.READ)

    const stagingData = new Int32Array(stagingBuffer.getMappedRange())
    const result = stagingData[0]
    stagingBuffer.unmap()

    const time = performance.now() - start

    return { result, time }
  }

  return sumReduction
}

const setupSumTile = async (input: Int32Array) => {
  const workgroupSize = Math.min(64, input.length)
  const workgroupsPerTile = Math.min(32, input.length / workgroupSize)
  const tileSize = workgroupSize * workgroupsPerTile

  let workgroupCountX = input.length / tileSize
  let workgroupCountY = 1
  while (workgroupCountX > 65535) {
    workgroupCountX /= 2
    workgroupCountY *= 2
  }

  const clearModule = device.createShaderModule({
    code: /* wgsl */ `
      @group(0) @binding(0)
      var<storage, read_write> output: u32;

      @compute @workgroup_size(1)
      fn main() {
        output = 0u;
      }
    `,
  })

  const sumModule = device.createShaderModule({
    code: /* wgsl */ `
      @group(0) @binding(0)
      var<storage> input: array<u32>;

      @group(0) @binding(1)
      var<storage, read_write> output: atomic<u32>;

      var<workgroup> sharedData: array<u32, ${workgroupSize}>;

      @compute @workgroup_size(${workgroupSize})
      fn main(
        @builtin(workgroup_id)
        workgroupId: vec3u,

        @builtin(local_invocation_index)
        localIndex: u32,
      ) {
        let workgroupIndex = workgroupId.x + workgroupId.y * ${workgroupCountX}u;
        let i = workgroupIndex * ${tileSize}u + localIndex;

        var sum = 0u;
        for (var j = 0u; j < ${workgroupsPerTile}u; j++) {
          sum += input[i + j * ${workgroupSize}u];
        }
        sharedData[localIndex] = sum;
        workgroupBarrier();

        for (var stride = ${workgroupSize}u / 2u; stride > 0u; stride >>= 1u) {
          if (localIndex < stride) {
            sharedData[localIndex] += sharedData[localIndex + stride];
          }
          workgroupBarrier();
        }

        if (localIndex == 0u) {
          atomicAdd(&output, sharedData[0]);
        }
      }
    `,
  })

  const inputBuffer = device.createBuffer({
    size: input.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(inputBuffer, 0, input)

  const outputBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  })
  const stagingBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  })

  const clearBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
    ],
  })
  const clearBindGroup = device.createBindGroup({
    layout: clearBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: outputBuffer } }],
  })
  const clearPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [clearBindGroupLayout],
  })
  const clearPipeline = device.createComputePipeline({
    layout: clearPipelineLayout,
    compute: { module: clearModule, entryPoint: 'main' },
  })

  const sumBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
    ],
  })
  const sumBindGroup = device.createBindGroup({
    layout: sumBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
    ],
  })
  const sumPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [sumBindGroupLayout],
  })
  const sumPipeline = device.createComputePipeline({
    layout: sumPipelineLayout,
    compute: { module: sumModule, entryPoint: 'main' },
  })

  const sumTile = async (): Promise<SumResult> => {
    const start = performance.now()

    const encoder = device.createCommandEncoder()

    const clearPass = encoder.beginComputePass()
    clearPass.setPipeline(clearPipeline)
    clearPass.setBindGroup(0, clearBindGroup)
    clearPass.dispatchWorkgroups(1)
    clearPass.end()

    const sumPass = encoder.beginComputePass()
    sumPass.setPipeline(sumPipeline)
    sumPass.setBindGroup(0, sumBindGroup)
    sumPass.dispatchWorkgroups(workgroupCountX, workgroupCountY)
    sumPass.end()

    encoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, 4)

    const commands = encoder.finish()
    device.queue.submit([commands])

    await stagingBuffer.mapAsync(GPUMapMode.READ)

    const stagingData = new Int32Array(stagingBuffer.getMappedRange())
    const result = stagingData[0]
    stagingBuffer.unmap()

    const time = performance.now() - start

    return { result, time }
  }

  return sumTile
}

const setupSumVec = async (input: Int32Array) => {
  const workgroupSize = Math.min(64, input.length / 4)
  const workgroupsPerTile = Math.min(8, input.length / 4 / workgroupSize)
  const tileSize = workgroupSize * workgroupsPerTile

  let workgroupCountX = input.length / tileSize / 4
  let workgroupCountY = 1
  while (workgroupCountX > 65535) {
    workgroupCountX /= 2
    workgroupCountY *= 2
  }

  const clearModule = device.createShaderModule({
    code: /* wgsl */ `
      @group(0) @binding(0)
      var<storage, read_write> output: u32;

      @compute @workgroup_size(1)
      fn main() {
        output = 0u;
      }
    `,
  })

  const sumModule = device.createShaderModule({
    code: /* wgsl */ `
      @group(0) @binding(0)
      var<storage> input: array<vec4u>;

      @group(0) @binding(1)
      var<storage, read_write> output: atomic<u32>;

      var<workgroup> sharedData: array<u32, ${workgroupSize}>;

      @compute @workgroup_size(${workgroupSize})
      fn main(
        @builtin(workgroup_id)
        workgroupId: vec3u,

        @builtin(local_invocation_index)
        localIndex: u32,
      ) {
        let workgroupIndex = workgroupId.x + workgroupId.y * ${workgroupCountX}u;
        let i = workgroupIndex * ${tileSize}u + localIndex;

        var sum = 0u;
        for (var j = 0u; j < ${workgroupsPerTile}u; j++) {
          let value = input[i + j * ${workgroupSize}u];
          sum += value.x + value.y + value.z + value.w;
        }
        sharedData[localIndex] = sum;
        workgroupBarrier();

        for (var stride = ${workgroupSize}u / 2u; stride > 0u; stride >>= 1u) {
          if (localIndex < stride) {
            sharedData[localIndex] += sharedData[localIndex + stride];
          }
          workgroupBarrier();
        }

        if (localIndex == 0u) {
          atomicAdd(&output, sharedData[0]);
        }
      }
    `,
  })

  const inputBuffer = device.createBuffer({
    size: input.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(inputBuffer, 0, input)

  const outputBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  })
  const stagingBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  })

  const clearBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
    ],
  })
  const clearBindGroup = device.createBindGroup({
    layout: clearBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: outputBuffer } }],
  })
  const clearPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [clearBindGroupLayout],
  })
  const clearPipeline = device.createComputePipeline({
    layout: clearPipelineLayout,
    compute: { module: clearModule, entryPoint: 'main' },
  })

  const sumBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
    ],
  })
  const sumBindGroup = device.createBindGroup({
    layout: sumBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
    ],
  })
  const sumPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [sumBindGroupLayout],
  })
  const sumPipeline = device.createComputePipeline({
    layout: sumPipelineLayout,
    compute: { module: sumModule, entryPoint: 'main' },
  })

  const sumVec = async (): Promise<SumResult> => {
    const start = performance.now()

    const encoder = device.createCommandEncoder()

    const clearPass = encoder.beginComputePass()
    clearPass.setPipeline(clearPipeline)
    clearPass.setBindGroup(0, clearBindGroup)
    clearPass.dispatchWorkgroups(1)
    clearPass.end()

    const sumPass = encoder.beginComputePass()
    sumPass.setPipeline(sumPipeline)
    sumPass.setBindGroup(0, sumBindGroup)
    sumPass.dispatchWorkgroups(workgroupCountX, workgroupCountY)
    sumPass.end()

    encoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, 4)

    const commands = encoder.finish()
    device.queue.submit([commands])

    await stagingBuffer.mapAsync(GPUMapMode.READ)

    const stagingData = new Int32Array(stagingBuffer.getMappedRange())
    const result = stagingData[0]
    stagingBuffer.unmap()

    const time = performance.now() - start

    return { result, time }
  }

  return sumVec
}

type SumPass = {
  module: GPUShaderModule
  workgroupCount: number
  workgroupCountX: number
  workgroupCountY: number
}

const setupSumMulti = async (input: Int32Array) => {
  const maxWorkgroupSize = 64
  const maxWorkgroupsPerTile = 32

  const passes: SumPass[] = []

  let workgroupCount = input.length
  while (workgroupCount > 1) {
    const workgroupSize = Math.min(maxWorkgroupSize, workgroupCount)
    const workgroupsPerTile = Math.min(
      maxWorkgroupsPerTile,
      workgroupCount / workgroupSize,
    )
    const tileSize = workgroupSize * workgroupsPerTile

    let workgroupCountX = workgroupCount / tileSize
    let workgroupCountY = 1
    while (workgroupCountX > 65535) {
      workgroupCountX /= 2
      workgroupCountY *= 2
    }
    workgroupCount = workgroupCountX * workgroupCountY

    const module = device.createShaderModule({
      code: /* wgsl */ `
        @group(0) @binding(0)
        var<storage> input: array<u32>;

        @group(0) @binding(1)
        var<storage, read_write> output: array<u32>;

        var<workgroup> sharedData: array<u32, ${workgroupSize}>;

        @compute @workgroup_size(${workgroupSize})
        fn main(
          @builtin(workgroup_id)
          workgroupId: vec3u,

          @builtin(local_invocation_index)
          localIndex: u32,
        ) {
          let workgroupIndex = workgroupId.x + workgroupId.y * ${workgroupCountX}u;
          let i = workgroupIndex * ${tileSize}u + localIndex;

          var sum = 0u;
          for (var j = 0u; j < ${workgroupsPerTile}u; j++) {
            sum += input[i + j * ${workgroupSize}u];
          }
          sharedData[localIndex] = sum;
          workgroupBarrier();

          for (var stride = ${workgroupSize}u / 2u; stride > 0u; stride >>= 1u) {
            if (localIndex < stride) {
              sharedData[localIndex] += sharedData[localIndex + stride];
            }
            workgroupBarrier();
          }

          if (localIndex == 0u) {
            output[workgroupIndex] =  sharedData[0];
          }
        }
      `,
    })
    passes.push({ module, workgroupCount, workgroupCountX, workgroupCountY })
  }

  const lastPassBufferIndex = (passes.length - 1) % 2

  const inputBuffer = device.createBuffer({
    size: input.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(inputBuffer, 0, input)

  const outputBuffers = passes
    .slice(0, 2)
    .map(({ workgroupCount }, passIndex) =>
      device.createBuffer({
        size: workgroupCount * 4,
        usage:
          passIndex === lastPassBufferIndex
            ? GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            : GPUBufferUsage.STORAGE,
      }),
    )
  const stagingBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  })

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
    ],
  })
  const inputBindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: outputBuffers[0] } },
    ],
  })
  const outputBindGroups = Array.from(
    { length: Math.min(2, passes.length - 1) },
    (_, i) =>
      device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: outputBuffers[i] } },
          { binding: 1, resource: { buffer: outputBuffers[1 - i] } },
        ],
      }),
  )
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  })
  const pipelines = passes.map(({ module }) =>
    device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module, entryPoint: 'main' },
    }),
  )

  console.log({ passes, lastPassBufferIndex, outputBuffers })

  const sumMulti = async (): Promise<SumResult> => {
    const start = performance.now()

    const encoder = device.createCommandEncoder()

    passes.forEach(({ workgroupCountX, workgroupCountY }, passIndex) => {
      const pass = encoder.beginComputePass()
      pass.setPipeline(pipelines[passIndex])
      if (passIndex === 0) {
        pass.setBindGroup(0, inputBindGroup)
      } else {
        pass.setBindGroup(0, outputBindGroups[(passIndex - 1) % 2])
      }
      pass.dispatchWorkgroups(workgroupCountX, workgroupCountY)
      pass.end()
    })

    encoder.copyBufferToBuffer(
      outputBuffers[lastPassBufferIndex],
      0,
      stagingBuffer,
      0,
      4,
    )

    const commands = encoder.finish()
    device.queue.submit([commands])

    await stagingBuffer.mapAsync(GPUMapMode.READ)

    const stagingData = new Int32Array(stagingBuffer.getMappedRange())
    const result = stagingData[0]
    stagingBuffer.unmap()

    const time = performance.now() - start

    return { result, time }
  }

  return sumMulti
}

type RunLimitType = 'count' | 'duration'

const searchParams = new URLSearchParams(location.search)

const params = {
  minIntCount: Number(searchParams.get('minIntCount') ?? 2 ** 17),
  maxIntCount: Number(searchParams.get('maxIntCount') ?? 2 ** 25),
  runLimit: Number(searchParams.get('runLimit') ?? 500),
  runLimitType:
    searchParams.get('runLimitType') ?? ('duration' as RunLimitType),
}

type ParamName = keyof typeof params

const superscripts: Record<string, string> = {
  '0': '⁰',
  '1': '¹',
  '2': '²',
  '3': '³',
  '4': '⁴',
  '5': '⁵',
  '6': '⁶',
  '7': '⁷',
  '8': '⁸',
  '9': '⁹',
}

const formatIntCount = (count: number) => {
  return `${count} (2${[...String(Math.log2(count))]
    .map((value) => superscripts[value])
    .join('')})`
}

const populateIntCountOptions = (selectElement: HTMLSelectElement) => {
  for (let i = 2; i <= 26; i++) {
    const count = 2 ** i
    const option = document.createElement('option')
    option.value = String(count)
    option.textContent = formatIntCount(count)
    selectElement.appendChild(option)
  }
}

const bindElement = (
  element: HTMLInputElement | HTMLSelectElement,
  paramName: ParamName,
  callback?: () => void,
) => {
  const value = params[paramName]
  element.value = typeof value === 'string' ? value : String(value)
  element.onchange = () => {
    ;(params[paramName] as typeof value) =
      typeof value === 'string' ? element.value : Number(element.value)
    const searchParams = new URLSearchParams(location.search)
    searchParams.set(paramName, element.value)
    history.replaceState(null, '', `?${searchParams}`)
    callback?.()
  }
}

const waitForUIUpdate = () => new Promise((resolve) => setTimeout(resolve, 10))

const setups = [
  setupSumCPU,
  setupSumReduction,
  setupSumTile,
  setupSumVec,
  setupSumMulti,
]

const initTable = () => {
  const { minIntCount, maxIntCount } = params

  const intCountHeader = document.querySelector<HTMLTableCellElement>(
    'thead tr:first-of-type th:last-of-type',
  )!
  intCountHeader.colSpan = Math.log2(maxIntCount) - Math.log2(minIntCount) + 1

  const intCountValues = document.querySelector<HTMLTableRowElement>(
    'thead tr:last-of-type',
  )!
  const intCountValuesChildren: HTMLTableCellElement[] = []
  intCountValuesChildren.push(document.createElement('th'))
  for (let count = minIntCount; count <= maxIntCount; count *= 2) {
    const headerCell = document.createElement('th')
    headerCell.textContent = formatIntCount(count)
    intCountValuesChildren.push(headerCell)
  }
  intCountValues.replaceChildren(...intCountValuesChildren)

  const tableBody = document.querySelector('tbody')!
  const tableRows: HTMLTableRowElement[] = []
  for (const setupSum of setups) {
    const sumName = setupSum.name.replace('setupS', 's')

    const row = document.createElement('tr')

    const dataCell = document.createElement('td')
    dataCell.textContent = sumName
    row.appendChild(dataCell)

    for (let count = minIntCount; count <= maxIntCount; count *= 2) {
      const dataCell = document.createElement('td')
      dataCell.id = `${sumName}-${count}`
      dataCell.textContent = 'ready'
      row.appendChild(dataCell)
    }
    tableRows.push(row)
  }
  tableBody.replaceChildren(...tableRows)
}

initTable()

const minIntCountElement =
  document.querySelector<HTMLSelectElement>('[name=minIntCount]')!
populateIntCountOptions(minIntCountElement)
bindElement(minIntCountElement, 'minIntCount', initTable)

const maxIntCountElement =
  document.querySelector<HTMLSelectElement>('[name=maxIntCount]')!
populateIntCountOptions(maxIntCountElement)
bindElement(maxIntCountElement, 'maxIntCount', initTable)

bindElement(
  document.querySelector<HTMLInputElement>('[name=runLimit]')!,
  'runLimit',
)
bindElement(
  document.querySelector<HTMLSelectElement>('[name=runLimitType')!,
  'runLimitType',
)

document.querySelector('button')!.onclick = async () => {
  document.querySelectorAll('td[id]').forEach((cell) => {
    cell.className = ''
    cell.textContent = 'pending...'
  })
  await waitForUIUpdate()

  const { minIntCount, maxIntCount, runLimit, runLimitType } = params
  for (let count = minIntCount; count <= maxIntCount; count *= 2) {
    console.groupCollapsed(`${formatIntCount(count)} ints`)

    let firstResult: number | undefined
    const input = generateInput(count)
    const means: Record<string, number> = {}
    for (const setupSum of setups) {
      const sum = await setupSum(input)

      const id = `${sum.name}-${count}`

      const cell = document.getElementById(id)!
      cell.textContent = 'running...'
      await waitForUIUpdate()

      console.groupCollapsed(sum.name)
      let latestResult: number | undefined
      let timeSum = 0
      let runCount = 0
      if (runLimitType === 'count') {
        for (let i = 0; i < runLimit; i++) {
          const { result, time } = await sum()
          console.log(`time: ${time} ms`)
          latestResult = result
          timeSum += time
          runCount++
        }
      } else {
        const start = performance.now()
        while (performance.now() - start < runLimit) {
          const { result, time } = await sum()
          console.log(`time: ${time} ms`)
          latestResult = result
          timeSum += time
          runCount++
        }
      }
      console.log(`result: ${latestResult}`)
      console.groupEnd()

      if (firstResult === undefined) {
        firstResult = latestResult
      }
      if (latestResult === firstResult) {
        means[`${sum.name}-${count}`] = timeSum / runCount
        cell.textContent = 'completed'
      } else {
        cell.textContent = 'ERROR'
        cell.classList.add('result__cell--error')
      }
      await waitForUIUpdate()
    }
    console.groupEnd()

    let fastest = ''
    let fastestTime = Infinity
    let slowest = ''
    let slowestTime = -Infinity
    for (const [id, mean] of Object.entries(means)) {
      if (mean === undefined) {
        continue
      }
      if (mean < fastestTime) {
        fastest = id
        fastestTime = mean
      }
      if (mean > slowestTime) {
        slowest = id
        slowestTime = mean
      }
      document.getElementById(id)!.textContent = `${mean.toLocaleString(
        'en-US',
        { maximumFractionDigits: 3 },
      )} ms`
    }
    if (fastest) {
      document.getElementById(fastest)!.classList.add('result__cell--fastest')
    }
    if (slowest) {
      document.getElementById(slowest)!.classList.add('result__cell--slowest')
    }
    await waitForUIUpdate()
  }
}
