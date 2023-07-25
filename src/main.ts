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

const layer1Bias = await fetchParameter('layer1-bias')
console.log(
  `Layer 1 bias: ${[...layer1Bias.slice(0, 3)]
    .map((value) => value.toFixed(7))
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
  
  uniform sampler2D xTex;
  uniform sampler2D weightTex;

  out float result;

  void main() {
    ivec2 fragCoord = ivec2(gl_FragCoord);
    ivec2 weightCoord = ivec2(fragCoord.x * fragCoord.y, 0);
    float x = texelFetch(xTex, fragCoord, 0).r;
    float weight = texelFetch(weightTex, weightCoord, 0).r;
    result = x * weight;
  }
`

const programInfo = twgl.createProgramInfo(gl, [vertexShader, fragmentShader])

const arrays = {
  inPosition: [-1, -1, 0, 1, -1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1, 1, 0],
}
const bufferInfo = twgl.createBufferInfoFromArrays(gl, arrays)

type Textures = {
  x: WebGLTexture
  weight: WebGLTexture
}

const textures = await new Promise<Textures>((resolve) => {
  const result = twgl.createTextures(
    gl,
    {
      x: { src: 'data/fashion-mnist/0.png' },
      weight: {
        src: layer1Weight,
        internalFormat: gl.R32F,
        width: 784,
        height: 512,
      },
    },
    () => resolve(result),
  ) as Textures
})

const frameBufferInfo = twgl.createFramebufferInfo(
  gl,
  [{ internalFormat: gl.R32F }],
  28,
  28,
)

gl.viewport(0, 0, 28, 28)

const uniforms = {
  xTex: textures.x,
  weightTex: textures.weight,
}

gl.bindFramebuffer(gl.FRAMEBUFFER, frameBufferInfo.framebuffer)
gl.useProgram(programInfo.program)
twgl.setBuffersAndAttributes(gl, programInfo, bufferInfo)
twgl.setUniforms(programInfo, uniforms)
twgl.drawBufferInfo(gl, bufferInfo)

const y = new Float32Array(784)
gl.readPixels(0, 0, 28, 28, gl.RED, gl.FLOAT, y)
console.log(`y: ${y.join(', ')}`)
