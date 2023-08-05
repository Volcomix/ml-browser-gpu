import './style.css'

if (!navigator.gpu) {
  throw Error('WebGPU not supported.')
}
const adapter = await navigator.gpu.requestAdapter()
if (!adapter) {
  throw Error("Couldn't request WebGPU adapter.")
}
const device = await adapter.requestDevice()

const time = (label = 'time') => console.time(label)
const timeEnd = (label = 'time') => console.timeEnd(label)

const input = new Int32Array(2 ** 22)

const populateInput = () => {
  time(populateInput.name)
  for (let i = 0; i < input.length; i++) {
    input[i] = Math.round(Math.random() * 10)
  }
  timeEnd(populateInput.name)
}

const sumCPU = () => {
  time()
  const result = input.reduce((a, b) => a + b)
  timeEnd()
  return result
}

const setupSumSequential = async () => {
  const module = device.createShaderModule({
    code: /* wgsl */ `
      @group(0) @binding(0)
      var<storage> input: array<i32>;

      @group(0) @binding(1)
      var<storage, read_write> output: array<i32>;

      @compute @workgroup_size(1)
      fn main() {
        var sum = 0i;
        for (var i = 0i; i < ${input.length}; i++) {
          sum += input[i];
        }
        output[0] = sum;
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
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
    ],
  })
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  })
  const pipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module, entryPoint: 'main' },
  })

  const sumSequential = async () => {
    time()

    const encoder = device.createCommandEncoder()

    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(1)
    pass.end()

    encoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, 4)

    const commands = encoder.finish()
    device.queue.submit([commands])

    await stagingBuffer.mapAsync(GPUMapMode.READ)

    const stagingData = new Int32Array(stagingBuffer.getMappedRange())
    const result = stagingData[0]
    stagingBuffer.unmap()

    timeEnd()

    return result
  }

  return sumSequential
}

populateInput()
for (const sum of [sumCPU, await setupSumSequential()]) {
  console.group(sum.name)
  let result
  for (let i = 0; i < 10; i++) {
    result = await sum()
  }
  console.log(`result: ${result}`)
  console.groupEnd()
}
