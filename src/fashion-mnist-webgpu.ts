import './style.css'

if (!navigator.gpu) {
  throw Error('WebGPU not supported.')
}

const adapter = await navigator.gpu.requestAdapter()
if (!adapter) {
  throw Error("Couldn't request WebGPU adapter.")
}

const device = await adapter.requestDevice()

const loadParameter = async (name: string) => {
  const response = await fetch(`models/fashion-mnist/${name}.gz`)
  const arrayBuffer = await response.arrayBuffer()

  const buffer = device.createBuffer({
    size: arrayBuffer.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(buffer, 0, arrayBuffer)

  return buffer
}

const [weight0] = await Promise.all([loadParameter('0-weight')])

const result = device.createBuffer({
  size: weight0.size,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
})

const stagingBuffer = device.createBuffer({
  size: 8 * 4,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
})

const module = device.createShaderModule({
  code: /* wgsl */ `
    @group(0) @binding(0)
    var<storage> weight: array<f32>;

    @group(1) @binding(0)
    var<storage, read_write> result: array<f32>;

    @compute @workgroup_size(1)
    fn main(
      @builtin(global_invocation_id)
      id: vec3u
    ) {
      result[id.x] = weight[id.x];
    }
  `,
})

const parameterBindGroupLayout = device.createBindGroupLayout({
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'read-only-storage' },
    },
  ],
})

const parameterBindGroup = device.createBindGroup({
  layout: parameterBindGroupLayout,
  entries: [{ binding: 0, resource: { buffer: weight0 } }],
})

const resultBindGroupLayout = device.createBindGroupLayout({
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'storage' },
    },
  ],
})

const resultBindGroup = device.createBindGroup({
  layout: resultBindGroupLayout,
  entries: [{ binding: 0, resource: { buffer: result } }],
})

const pipelineLayout = device.createPipelineLayout({
  bindGroupLayouts: [parameterBindGroupLayout, resultBindGroupLayout],
})

const pipeline = device.createComputePipeline({
  layout: pipelineLayout,
  compute: {
    module,
    entryPoint: 'main',
  },
})

const encoder = device.createCommandEncoder()

const pass = encoder.beginComputePass()
pass.setPipeline(pipeline)
pass.setBindGroup(0, parameterBindGroup)
pass.setBindGroup(1, resultBindGroup)
pass.dispatchWorkgroups(4)
pass.end()

encoder.copyBufferToBuffer(result, 0, stagingBuffer, 0, 8 * 4)

const commands = encoder.finish()
device.queue.submit([commands])

await stagingBuffer.mapAsync(GPUMapMode.READ)
console.log(`Result: ${new Float32Array(stagingBuffer.getMappedRange())}`)

stagingBuffer.unmap()
