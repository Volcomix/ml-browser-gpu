import { VecMatMulResult } from './types'

if (!navigator.gpu) {
  throw Error('WebGPU not supported.')
}
const adapter = await navigator.gpu.requestAdapter()
if (!adapter) {
  throw Error("Couldn't request WebGPU adapter.")
}
const device = await adapter.requestDevice()

export const adapterInfo = await adapter.requestAdapterInfo()

type Buffers = {
  x: GPUBuffer
  a: GPUBuffer
  y: GPUBuffer
  staging: GPUBuffer
}

const loadInputBuffer = (data: Float32Array) => {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(buffer, 0, data)
  return buffer
}

const loadInputBuffers = (x: Float32Array, a: Float32Array) => ({
  x: loadInputBuffer(x),
  a: loadInputBuffer(a),
})

const createOutputBuffers = (yLength: number) => {
  const size = yLength * 4
  return {
    y: device.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    }),
    staging: device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    }),
  }
}

const createBuffers = (x: Float32Array, a: Float32Array, yLength: number) => ({
  ...loadInputBuffers(x, a),
  ...createOutputBuffers(yLength),
})

const createBindGroupLayout = () => {
  return device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
    ],
  })
}

const createBindGroup = (layout: GPUBindGroupLayout, buffers: Buffers) => {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: { buffer: buffers.x } },
      { binding: 1, resource: { buffer: buffers.a } },
      { binding: 2, resource: { buffer: buffers.y } },
    ],
  })
}

const createPipelineLayout = (bindGroupLayout: GPUBindGroupLayout) => {
  return device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] })
}

const createPipeline = (layout: GPUPipelineLayout, module: GPUShaderModule) => {
  return device.createComputePipeline({
    layout,
    compute: { module, entryPoint: 'main' },
  })
}

export const setupVecMatMulWebGPUSimple = (
  x: Float32Array,
  a: Float32Array,
) => {
  const workgroupSize = 64

  const module = device.createShaderModule({
    code: /* wgsl */ `
      @group(0) @binding(0)
      var<storage> x: array<f32>;

      @group(0) @binding(1)
      var<storage> a: array<f32>;

      @group(0) @binding(2)
      var<storage, read_write> y: array<f32>;

      @compute @workgroup_size(${workgroupSize})
      fn main(
        @builtin(global_invocation_id)
        globalInvocationId: vec3u,
      ) {
        let row = globalInvocationId.x;
        var sum = 0f;
        for (var i = 0u; i < ${x.length}u; i++) {
          sum += a[row * ${x.length}u + i] * x[i];
        }
        y[row] = sum;
      }
    `,
  })

  const yLength = a.length / x.length
  const buffers = createBuffers(x, a, yLength)
  const bindGroupLayout = createBindGroupLayout()
  const bindGroup = createBindGroup(bindGroupLayout, buffers)
  const pipelineLayout = createPipelineLayout(bindGroupLayout)
  const pipeline = createPipeline(pipelineLayout, module)

  const result = new Float32Array(yLength)

  const vecMatMulWebGPUSimple = async (): Promise<VecMatMulResult> => {
    const start = performance.now()

    const encoder = device.createCommandEncoder()

    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(yLength / workgroupSize)
    pass.end()

    encoder.copyBufferToBuffer(buffers.y, 0, buffers.staging, 0, buffers.y.size)

    const commands = encoder.finish()
    device.queue.submit([commands])

    await buffers.staging.mapAsync(GPUMapMode.READ)

    const arrayBuffer = new Float32Array(buffers.staging.getMappedRange())
    result.set(arrayBuffer, 0)
    buffers.staging.unmap()

    const time = performance.now() - start

    return { result, time }
  }

  return vecMatMulWebGPUSimple
}
