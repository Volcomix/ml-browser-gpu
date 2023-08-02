import './style.css'

if (!navigator.gpu) {
  throw Error('WebGPU not supported.')
}

const adapter = await navigator.gpu.requestAdapter()
if (!adapter) {
  throw Error("Couldn't request WebGPU adapter.")
}

const device = await adapter.requestDevice()

const count = 4
const bufferSize = count * 4

const module = device.createShaderModule({
  code: /* wgsl */ `
    @group(0) @binding(0)
    var<storage, read_write> data: array<f32>;

    @compute @workgroup_size(1)
    fn computeSomething(
      @builtin(global_invocation_id)
      id: vec3u
    ) {
      data[id.x] = f32(id.x) * 2.0;
    }
  `,
})

const pipeline = device.createComputePipeline({
  layout: 'auto',
  compute: {
    module,
    entryPoint: 'computeSomething',
  },
})

const output = device.createBuffer({
  size: bufferSize,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
})

const stagingBuffer = device.createBuffer({
  size: bufferSize,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
})

const bindGroup = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [{ binding: 0, resource: { buffer: output } }],
})

const encoder = device.createCommandEncoder()

const pass = encoder.beginComputePass()
pass.setPipeline(pipeline)
pass.setBindGroup(0, bindGroup)
pass.dispatchWorkgroups(count)
pass.end()

encoder.copyBufferToBuffer(output, 0, stagingBuffer, 0, bufferSize)

const commands = encoder.finish()
device.queue.submit([commands])

await stagingBuffer.mapAsync(GPUMapMode.READ)
const result = new Float32Array(stagingBuffer.getMappedRange())

console.log(`Result: ${result}`)

stagingBuffer.unmap()
