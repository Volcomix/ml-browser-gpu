import * as twgl from 'twgl.js'

import './style.css'

const fetchParameter = async (name: string) => {
  const response = await fetch(`models/fashion-mnist/${name}.gz`)
  const buffer = await response.arrayBuffer()
  return new Float32Array(buffer)
}

const layer1Weight = await fetchParameter('layer1-weight')
console.log(
  `Layer 1 weight: ${[...layer1Weight.slice(0, 3)]
    .map((value) => value.toFixed(4))
    .join(', ')}, ...`,
)

const canvas = new OffscreenCanvas(1, 1)
const gl = canvas.getContext('webgl2')
if (!gl) {
  throw new Error('WebGL 2 is not available')
}
twgl.addExtensionsToContext(gl)

const vertexShader = /* glsl */ `#version 300 es
     
  in vec2 inPosition;

  void main() {
    gl_Position = vec4(inPosition, 0, 1);
  }
`

const fragmentShader = /* glsl */ `#version 300 es

  precision highp float;
  
  uniform sampler2D x;

  out float result;

  void main() {
    result = texelFetch(x, ivec2(gl_FragCoord), 0).r;
  }
`

const programInfo = twgl.createProgramInfo(gl, [vertexShader, fragmentShader])

const arrays = {
  inPosition: [-1, -1, 0, 1, -1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1, 1, 0],
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
console.log(`y: ${y.join(', ')}`)
