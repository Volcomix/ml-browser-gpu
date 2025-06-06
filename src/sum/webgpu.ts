import { SumResult } from './types'

if (!navigator.gpu) {
  throw Error('WebGPU not supported.')
}
const adapter = await navigator.gpu.requestAdapter()
if (!adapter) {
  throw Error("Couldn't request WebGPU adapter.")
}
const device = await adapter.requestDevice({ requiredFeatures: ['subgroups'] })

export const adapterInfo = adapter.info

export const setupSumWebGPUAtomic = (input: Uint32Array) => {
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

  const sumWebGPUAtomic = async (): Promise<SumResult> => {
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

    const stagingData = new Uint32Array(stagingBuffer.getMappedRange())
    const result = stagingData[0]
    stagingBuffer.unmap()

    const time = performance.now() - start

    return { result, time }
  }

  return sumWebGPUAtomic
}

export const setupSumWebGPUTile = (input: Uint32Array) => {
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

  const sumWebGPUTile = async (): Promise<SumResult> => {
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

    const stagingData = new Uint32Array(stagingBuffer.getMappedRange())
    const result = stagingData[0]
    stagingBuffer.unmap()

    const time = performance.now() - start

    return { result, time }
  }

  return sumWebGPUTile
}

export const setupSumWebGPUVector = (input: Uint32Array) => {
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

  const sumWebGPUVector = async (): Promise<SumResult> => {
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

    const stagingData = new Uint32Array(stagingBuffer.getMappedRange())
    const result = stagingData[0]
    stagingBuffer.unmap()

    const time = performance.now() - start

    return { result, time }
  }

  return sumWebGPUVector
}

type SumPass = {
  module: GPUShaderModule
  workgroupCount: number
  workgroupCountX: number
  workgroupCountY: number
}

export const setupSumWebGPURecursive = (input: Uint32Array) => {
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

  const sumWebGPURecursive = async (): Promise<SumResult> => {
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

    const stagingData = new Uint32Array(stagingBuffer.getMappedRange())
    const result = stagingData[0]
    stagingBuffer.unmap()

    const time = performance.now() - start

    return { result, time }
  }

  return sumWebGPURecursive
}

export const setupSumWebGPUSubgroup = (input: Uint32Array) => {
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
      enable subgroups;

      @group(0) @binding(0)
      var<storage> input: array<u32>;

      @group(0) @binding(1)
      var<storage, read_write> output: atomic<u32>;

      @compute @workgroup_size(${workgroupSize})
      fn main(
        @builtin(global_invocation_id)
        globalId: vec3u,
      ) {
        let i = globalId.x + globalId.y * ${workgroupCountX * workgroupSize};
        let sum = subgroupAdd(input[i]);

        if (subgroupElect()) {
          atomicAdd(&output, sum);
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

  const sumWebGPUSubgroup = async (): Promise<SumResult> => {
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

    const stagingData = new Uint32Array(stagingBuffer.getMappedRange())
    const result = stagingData[0]
    stagingBuffer.unmap()

    const time = performance.now() - start

    return { result, time }
  }

  return sumWebGPUSubgroup
}

export const setupSumWebGPUSubgroupTile = (input: Uint32Array) => {
  const workgroupSize = Math.min(64, input.length)
  const tileSize = Math.min(32, input.length / workgroupSize)
  const gridSize = input.length / tileSize

  let workgroupCountX = gridSize / workgroupSize
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
      enable subgroups;

      @group(0) @binding(0)
      var<storage> input: array<u32>;

      @group(0) @binding(1)
      var<storage, read_write> output: atomic<u32>;

      @compute @workgroup_size(${workgroupSize})
      fn main(
        @builtin(global_invocation_id)
        globalId: vec3u,
      ) {
        var i = globalId.x + globalId.y * ${workgroupCountX * workgroupSize};

        var sum = 0u;
        for (; i < ${input.length}u; i += ${gridSize}u) {
          sum += input[i];
        }
        
        sum = subgroupAdd(sum);

        if (subgroupElect()) {
          atomicAdd(&output, sum);
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

  const sumWebGPUSubgroupTile = async (): Promise<SumResult> => {
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

    const stagingData = new Uint32Array(stagingBuffer.getMappedRange())
    const result = stagingData[0]
    stagingBuffer.unmap()

    const time = performance.now() - start

    return { result, time }
  }

  return sumWebGPUSubgroupTile
}
