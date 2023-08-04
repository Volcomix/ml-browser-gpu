import * as twgl from 'twgl.js'

import './style.css'

const dimensionSize = 2048

const width = dimensionSize
const height = dimensionSize
const data = new Float32Array(width * height)
const resultData = new Float32Array(4)

const populateData = () => {
  const start = performance.now()
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.random()
  }
  const elapsed = performance.now() - start

  console.log(`Data populated in ${elapsed}ms`)
}

const sumJs = () => {
  const start = performance.now()
  const result = data.reduce((p, c) => p + c)
  const elapsed = performance.now() - start

  console.log(`[js] Result: ${result}`)
  console.log(`[js] Elapsed: ${elapsed}ms`)
}

const setupWebGL = () => {
  const canvas = new OffscreenCanvas(1, 1)
  const gl = canvas.getContext('webgl2')
  if (!gl) {
    throw new Error('WebGL 2 is not available')
  }
  twgl.addExtensionsToContext(gl)

  const bufferInfo = twgl.createBufferInfoFromArrays(gl, {
    inPosition: [-1, -1, 0, 1, -1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1, 1, 0],
  })

  const vertexShader = /* glsl */ `#version 300 es
      
    in vec2 inPosition;

    void main() {
      gl_Position = vec4(inPosition, 0, 1);
    }
  `

  const lod = Math.log2(dimensionSize)

  const fragmentShader = /* glsl */ `#version 300 es

    precision highp float;

    uniform sampler2D xTex;

    out float y;

    void main() {
      y = texelFetch(xTex, ivec2(gl_FragCoord), ${lod}).r * ${data.length}.0;
    }
  `

  const programInfo = twgl.createProgramInfo(gl, [vertexShader, fragmentShader])

  const texture = twgl.createTexture(gl, {
    internalFormat: gl.R32F,
    src: data,
    width,
    height,
  })

  const fbi = twgl.createFramebufferInfo(
    gl,
    [{ internalFormat: gl.R32F }],
    1,
    1,
  )

  const sumWebGL = () => {
    const start = performance.now()

    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.generateMipmap(gl.TEXTURE_2D)

    gl.viewport(0, 0, 1, 1)

    gl.bindFramebuffer(gl.FRAMEBUFFER, fbi.framebuffer)
    gl.useProgram(programInfo.program)
    twgl.setBuffersAndAttributes(gl, programInfo, bufferInfo)
    twgl.setUniforms(programInfo, { xTex: texture })
    twgl.drawBufferInfo(gl, bufferInfo)

    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, resultData)

    const elapsed = performance.now() - start

    console.log(`[WebGL] Result: ${resultData[0]}`)
    console.log(`[WebGL] Elapsed: ${elapsed}ms`)
  }

  return sumWebGL
}

const setupWebGPU = async () => {
  if (!navigator.gpu) {
    throw Error('WebGPU not supported.')
  }
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) {
    throw Error("Couldn't request WebGPU adapter.")
  }
  const device = await adapter.requestDevice()

  const module = device.createShaderModule({
    code: /* wgsl */ `
      @group(0) @binding(0)
      var<storage, read_write> data: array<f32>;
  
      @compute @workgroup_size(64)
      fn main(
        @builtin(global_invocation_id)
        global_id: vec3u,

        @builtin(local_invocation_id)
        local_id: vec3u
      ) {
        data[global_id.x] = data[global_id.x] * 2.0;
      }
    `,
  })

  const dataBuffer = device.createBuffer({
    size: 16 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  })
  const data = new Float32Array(dataBuffer.getMappedRange())
  data.set([10, 1, 8, -1, 0, -2, 3, 5, -2, -3, 2, 7, 0, 11, 0, 2])
  dataBuffer.unmap()

  const stagingBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  })

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
    ],
  })
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: dataBuffer } }],
  })
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  })
  const pipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module, entryPoint: 'main' },
  })

  const sumWebGPU = async () => {
    const start = performance.now()

    const encoder = device.createCommandEncoder()

    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(Math.ceil(dataBuffer.size / 4 / 64))
    pass.end()

    encoder.copyBufferToBuffer(dataBuffer, 0, stagingBuffer, 0, 4)

    const commands = encoder.finish()
    device.queue.submit([commands])

    await stagingBuffer.mapAsync(GPUMapMode.READ)

    const elapsed = performance.now() - start
    const stagingData = new Float32Array(stagingBuffer.getMappedRange())

    console.log(`[WebGPU] Elapsed: ${elapsed}ms`)
    console.log(`[WebGPU] Result: ${stagingData}`)

    stagingBuffer.unmap()
  }

  return sumWebGPU
}

populateData()

console.log('-'.repeat(40))

for (let i = 0; i < 10; i++) {
  sumJs()
}

console.log('-'.repeat(40))

const sumWebGL = setupWebGL()
for (let i = 0; i < 10; i++) {
  sumWebGL()
}

console.log('-'.repeat(40))

const sumWebGPU = await setupWebGPU()
for (let i = 0; i < 10; i++) {
  await sumWebGPU()
}
