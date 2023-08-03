import * as twgl from 'twgl.js'

import './style.css'

const dimensionSize = 2048
const width = dimensionSize
const height = dimensionSize
const data = new Float32Array(width * height)
const resultData = new Float32Array(1)

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

const sumWebGL = () => {
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

  const start = performance.now()

  gl.bindTexture(gl.TEXTURE_2D, texture)
  gl.generateMipmap(gl.TEXTURE_2D)

  gl.viewport(0, 0, 1, 1)

  gl.bindFramebuffer(gl.FRAMEBUFFER, fbi.framebuffer)
  gl.useProgram(programInfo.program)
  twgl.setBuffersAndAttributes(gl, programInfo, bufferInfo)
  twgl.setUniforms(programInfo, {
    xTex: texture,
  })
  twgl.drawBufferInfo(gl, bufferInfo)

  gl.readPixels(0, 0, 1, 1, gl.RED, gl.FLOAT, resultData)

  const elapsed = performance.now() - start

  console.log(`[WebGL] Result: ${resultData}`)
  console.log(`[WebGL] Elapsed: ${elapsed}ms`)
}

populateData()
console.log('-'.repeat(40))
sumJs()
console.log('-'.repeat(40))
sumWebGL()
