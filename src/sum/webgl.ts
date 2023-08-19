import * as twgl from 'twgl.js'

import { SumResult } from './types'

const canvas = new OffscreenCanvas(1, 1)
const gl = canvas.getContext('webgl2')
if (!gl) {
  throw new Error('WebGL 2 is not available')
}
twgl.addExtensionsToContext(gl)

export const setupSumWebGL = (input: Uint32Array) => {
  const idealDimension = Math.log2(Math.sqrt(input.length))
  const lod = Math.ceil(idealDimension)
  const width = 2 ** lod
  const height = 2 ** Math.floor(idealDimension)

  const vertexShader = /* glsl */ `#version 300 es
       
    in vec2 inPosition;
  
    void main() {
      gl_Position = vec4(inPosition, 0, 1);
    }
  `

  const fragmentShader = /* glsl */ `#version 300 es

    precision highp float;

    uniform sampler2D xTex;

    out uint y;

    void main() {
      y = uint(texelFetch(xTex, ivec2(gl_FragCoord), ${lod}).r * ${input.length}.0);
    }
  `

  const programInfo = twgl.createProgramInfo(gl, [vertexShader, fragmentShader])

  const bufferInfo = twgl.createBufferInfoFromArrays(gl, {
    inPosition: [-1, -1, 0, 1, -1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1, 1, 0],
  })
  const texture = twgl.createTexture(gl, {
    internalFormat: gl.R32F,
    src: new Float32Array(input),
    width,
    height,
  })
  const fbi = twgl.createFramebufferInfo(
    gl,
    [{ internalFormat: gl.R32UI }],
    1,
    1,
  )
  const output = new Uint32Array(4)

  const sumWebGL = (): SumResult => {
    const start = performance.now()

    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.generateMipmap(gl.TEXTURE_2D)

    gl.viewport(0, 0, 1, 1)

    gl.bindFramebuffer(gl.FRAMEBUFFER, fbi.framebuffer)
    gl.useProgram(programInfo.program)
    twgl.setBuffersAndAttributes(gl, programInfo, bufferInfo)
    twgl.setUniforms(programInfo, { xTex: texture })
    twgl.drawBufferInfo(gl, bufferInfo)

    gl.readPixels(0, 0, 1, 1, gl.RGBA_INTEGER, gl.UNSIGNED_INT, output)

    const time = performance.now() - start

    return { result: output[0], time }
  }

  return sumWebGL
}
