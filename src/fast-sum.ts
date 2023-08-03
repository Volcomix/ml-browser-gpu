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
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE,
  })

  console.log({ module, dataBuffer })

  const sumWebGPU = () => {}

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
  sumWebGPU()
}
