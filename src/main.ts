import * as twgl from 'twgl.js'

import './style.css'

const canvas = new OffscreenCanvas(1, 1)
const gl = canvas.getContext('webgl2')
if (!gl) {
  throw new Error('WebGL 2 is not available')
}
twgl.addExtensionsToContext(gl)

const vertexShader = /* glsl */ `#version 300 es
     
  in vec2 inPosition;
  in vec2 inTexCoord;

  out vec2 texCoord;

  void main() {
    gl_Position = vec4(inPosition, 0, 1);
    texCoord = inTexCoord;
  }
`

const fragmentShader = /* glsl */ `#version 300 es

  precision highp float;
  
  uniform sampler2D x;

  in vec2 texCoord;

  out float result;

  void main() {
    result = texture(x, texCoord).r; // TODO Use texelFetch instead
  }
`

const programInfo = twgl.createProgramInfo(gl, [vertexShader, fragmentShader])

const arrays = {
  inPosition: [-1, -1, 0, 1, -1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1, 1, 0],
  inTexCoord: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],
}
const bufferInfo = twgl.createBufferInfoFromArrays(gl, arrays)

const textures = twgl.createTextures(gl, {
  x: { internalFormat: gl.R32F, src: [1, 2, 3, 4, 5] },
})

const frameBufferInfo = twgl.createFramebufferInfo(
  gl,
  [{ internalFormat: gl.R32F }],
  5,
  1,
)

gl.viewport(0, 0, 5, 1)

const uniforms = {
  x: textures.x,
}

gl.bindFramebuffer(gl.FRAMEBUFFER, frameBufferInfo.framebuffer)
gl.useProgram(programInfo.program)
twgl.setBuffersAndAttributes(gl, programInfo, bufferInfo)
twgl.setUniforms(programInfo, uniforms)
twgl.drawBufferInfo(gl, bufferInfo)

const y = new Float32Array(5)
gl.readPixels(0, 0, 5, 1, gl.RED, gl.FLOAT, y)
console.log(`y: ${y}`)

// TODO Remove one of models/fashion-mnist/layer1-weight.bin
//                 or models/fashion-mnist/layer1-weight.gz
