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
    // TODO Rearrange the weights before uploading to texture
    ivec2 weightCoord = ivec2(
      (fragCoord.x % 32) + (fragCoord.y % 32) * 28,
      (fragCoord.x / 32) + (fragCoord.y / 32) * 23
    );
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
    float x = texelFetch(xTex, fragCoord, 5).r * 1024.0;
    float bias = texelFetch(biasTex, fragCoord, 0).r;
    y = x + bias; // TODO Fix result
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
  // TODO Put faces only on pixels where we want to compute something
  inPosition: [-1, -1, 0, 1, -1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1, 1, 0],
}
const bufferInfo = twgl.createBufferInfoFromArrays(gl, arrays)

// 32 = lowest power of 2 that is greater than or equal to 28
// 23 = ceil(sqrt(512))
// 529 = 23 * 23
// 736 = 32 * 23

// TODO Refactor this rearrangement
const layer1BiasReshaped = new Float32Array(529)
layer1BiasReshaped.set(layer1Bias)

type Textures = {
  x: WebGLTexture
  layer1Weight: WebGLTexture
  layer1Bias: WebGLTexture
}

const textures = await new Promise<Textures>((resolve) => {
  const result = twgl.createTextures(
    gl,
    {
      // TODO Consider grouping 4 successive R components into single RGBA pixel
      x: { src: 'data/fashion-mnist/0.png' },
      // TODO Set max mipmap level
      layer1Weight: {
        src: layer1Weight,
        internalFormat: gl.R32F,
        width: 784,
        height: 512,
      },
      layer1Bias: {
        src: layer1BiasReshaped,
        internalFormat: gl.R32F,
        width: 23,
        height: 23,
      },
    },
    () => resolve(result),
  ) as Textures
})

const layer1WeightFrameBufferInfo = twgl.createFramebufferInfo(
  gl,
  [{ internalFormat: gl.R32F }],
  736,
  736,
)

const layer1BiasFrameBufferInfo = twgl.createFramebufferInfo(
  gl,
  [{ internalFormat: gl.R32F }],
  23,
  23,
)

gl.viewport(0, 0, 736, 736)

const layer1WeightUniforms = {
  xTex: textures.x,
  weightTex: textures.layer1Weight,
}

gl.bindFramebuffer(gl.FRAMEBUFFER, layer1WeightFrameBufferInfo.framebuffer)
gl.useProgram(layer1WeightProgramInfo.program)
twgl.setBuffersAndAttributes(gl, layer1WeightProgramInfo, bufferInfo)
twgl.setUniforms(layer1WeightProgramInfo, layer1WeightUniforms)
twgl.drawBufferInfo(gl, bufferInfo)

gl.bindTexture(gl.TEXTURE_2D, layer1WeightFrameBufferInfo.attachments[0])
gl.generateMipmap(gl.TEXTURE_2D)

gl.viewport(0, 0, 23, 23)

const layer1BiasUniforms = {
  xTex: layer1WeightFrameBufferInfo.attachments[0],
  biasTex: textures.layer1Bias,
}

gl.bindFramebuffer(gl.FRAMEBUFFER, layer1BiasFrameBufferInfo.framebuffer)
gl.useProgram(layer1BiasProgramInfo.program)
twgl.setBuffersAndAttributes(gl, layer1BiasProgramInfo, bufferInfo)
twgl.setUniforms(layer1BiasProgramInfo, layer1BiasUniforms)
twgl.drawBufferInfo(gl, bufferInfo)

const hidden1 = new Float32Array(512)
gl.readPixels(0, 0, 23, 22, gl.RED, gl.FLOAT, hidden1)
gl.readPixels(0, 22, 6, 1, gl.RED, gl.FLOAT, hidden1, 506)
console.log(`Hidden 1: ${hidden1.join(', ')}`)
