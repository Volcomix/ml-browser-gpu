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

const layer1WeightFragmentShader = /* glsl */ `#version 300 es

  precision highp float;
  
  uniform sampler2D xTex;
  uniform sampler2D weightTex;

  out float y;

  void main() {
    ivec2 fragCoord = ivec2(gl_FragCoord);
    ivec2 weightCoord = ivec2(fragCoord.y * 28 + fragCoord.x, 0);
    float x = texelFetch(xTex, fragCoord, 0).r;
    float weight = texelFetch(weightTex, weightCoord, 0).r;
    y = x * weight;
  }
`

const layer1BiasFragmentShader = /* glsl */ `#version 300 es

  precision highp float;

  uniform sampler2D xTex;
  uniform sampler2D biasTex;

  out float y;

  void main() {
    ivec2 fragCoord = ivec2(gl_FragCoord);
    float x = texelFetch(xTex, fragCoord, 4).r * 784.0;
    float bias = texelFetch(biasTex, fragCoord, 0).r;
    y = x /* + bias */; // FIXME Wrong result
  }
`

const layer1WeightProgramInfo = twgl.createProgramInfo(gl, [
  vertexShader,
  layer1WeightFragmentShader,
])

const layer1BiasProgramInfo = twgl.createProgramInfo(gl, [
  vertexShader,
  layer1BiasFragmentShader,
])

const arrays = {
  inPosition: [-1, -1, 0, 1, -1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1, 1, 0],
}
const bufferInfo = twgl.createBufferInfoFromArrays(gl, arrays)

type Textures = {
  x: WebGLTexture
  layer1Weight: WebGLTexture
  layer1Bias: WebGLTexture
}

const textures = await new Promise<Textures>((resolve) => {
  const result = twgl.createTextures(
    gl,
    {
      x: { src: 'data/fashion-mnist/0.png' },
      layer1Weight: {
        src: layer1Weight,
        internalFormat: gl.R32F,
        width: 784,
        height: 512,
      },
      layer1Bias: {
        src: layer1Bias,
        internalFormat: gl.R32F,
        width: 512,
        height: 1,
      },
    },
    () => resolve(result),
  ) as Textures
})

const layer1WeightFrameBufferInfo = twgl.createFramebufferInfo(
  gl,
  [{ internalFormat: gl.R32F }],
  28,
  28,
)

const layer1BiasFrameBufferInfo = twgl.createFramebufferInfo(
  gl,
  [{ internalFormat: gl.R32F }],
  1,
  1,
)

gl.viewport(0, 0, 28, 28)

const layer1WeightUniforms = {
  xTex: textures.x,
  weightTex: textures.layer1Weight,
}

gl.bindFramebuffer(gl.FRAMEBUFFER, layer1WeightFrameBufferInfo.framebuffer)
gl.useProgram(layer1WeightProgramInfo.program)
twgl.setBuffersAndAttributes(gl, layer1WeightProgramInfo, bufferInfo)
twgl.setUniforms(layer1WeightProgramInfo, layer1WeightUniforms)
twgl.drawBufferInfo(gl, bufferInfo)

const layer1WeightY = new Float32Array(784)
gl.readPixels(0, 0, 28, 28, gl.RED, gl.FLOAT, layer1WeightY)
console.log(
  `Layer 1 weight y sum: ${layer1WeightY.reduce(
    (previous, current) => previous + current,
  )}`,
)

gl.bindTexture(gl.TEXTURE_2D, layer1WeightFrameBufferInfo.attachments[0])
gl.generateMipmap(gl.TEXTURE_2D)

gl.viewport(0, 0, 1, 1)

const layer1BiasUniforms = {
  xTex: layer1WeightFrameBufferInfo.attachments[0],
  biasTex: textures.layer1Bias,
}

gl.bindFramebuffer(gl.FRAMEBUFFER, layer1BiasFrameBufferInfo.framebuffer)
gl.useProgram(layer1BiasProgramInfo.program)
twgl.setBuffersAndAttributes(gl, layer1BiasProgramInfo, bufferInfo)
twgl.setUniforms(layer1BiasProgramInfo, layer1BiasUniforms)
twgl.drawBufferInfo(gl, bufferInfo)

const layer1BiasY = new Float32Array(1)
gl.readPixels(0, 0, 1, 1, gl.RED, gl.FLOAT, layer1BiasY)
console.log(`Hidden 1: ${layer1BiasY}`)
