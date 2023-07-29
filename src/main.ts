import * as twgl from 'twgl.js'

import './style.css'

const fetchParameter = async (name: string) => {
  const response = await fetch(`models/fashion-mnist/${name}.gz`)
  const buffer = await response.arrayBuffer()
  const result = new Float32Array(buffer)

  console.log(`${name}: ${[...result.slice(0, 3)].join(', ')}, ...`)

  return result
}

const canvas = new OffscreenCanvas(1, 1)
const gl = canvas.getContext('webgl2')
if (!gl) {
  throw new Error('WebGL 2 is not available')
}
twgl.addExtensionsToContext(gl)

const bufferInfo = twgl.createBufferInfoFromArrays(gl, {
  // TODO Check if possible optimizations by changing the faces
  inPosition: [-1, -1, 0, 1, -1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1, 1, 0],
})

const vertexShader = /* glsl */ `#version 300 es
     
  in vec2 inPosition;

  void main() {
    gl_Position = vec4(inPosition, 0, 1);
  }
`

const process = (
  programInfo: twgl.ProgramInfo,
  frameBufferInfo: twgl.FramebufferInfo,
  uniforms: Record<string, unknown>,
) => {
  gl.bindFramebuffer(gl.FRAMEBUFFER, frameBufferInfo.framebuffer)
  gl.useProgram(programInfo.program)
  twgl.setBuffersAndAttributes(gl, programInfo, bufferInfo)
  twgl.setUniforms(programInfo, uniforms)
  twgl.drawBufferInfo(gl, bufferInfo)
}

const loadWeight0 = async () => {
  const tex = twgl.createTexture(gl, {
    src: await fetchParameter('0-weight'),
    internalFormat: gl.R32F,
    width: 784,
    height: 512,
  })

  const fbi = twgl.createFramebufferInfo(
    gl,
    [{ internalFormat: gl.RGBA32F }],
    32,
    4096,
  )

  const fragmentShader = /* glsl */ `#version 300 es
  
    precision highp float;
  
    uniform sampler2D tex;
  
    out vec4 result;
  
    void main() {
      ivec2 fragCoord = ivec2(gl_FragCoord);
      int x = fragCoord.x + (fragCoord.y % 32) * 28;
      int y  = 4 * fragCoord.y / 32;
      result = vec4(
        texelFetch(tex, ivec2(x, y), 0).r,
        texelFetch(tex, ivec2(x, y + 1), 0).r,
        texelFetch(tex, ivec2(x, y + 2), 0).r,
        texelFetch(tex, ivec2(x, y + 3), 0).r
      );
    }
  `

  const programInfo = twgl.createProgramInfo(gl, [vertexShader, fragmentShader])
  gl.viewport(0, 0, 28, 4096)
  process(programInfo, fbi, { tex })

  gl.deleteTexture(tex)
  gl.deleteFramebuffer(fbi.framebuffer)
  gl.deleteProgram(programInfo.program)

  return fbi.attachments[0] as WebGLTexture
}

const loadBias0 = async () => {
  return twgl.createTexture(gl, {
    src: await fetchParameter('0-bias'),
    internalFormat: gl.RGBA32F,
    width: 1,
    height: 128,
  })
}

const weight0 = await loadWeight0()
const bias0 = await loadBias0()

console.log({ weight0, bias0 })

if (Number(1)) {
  throw new Error('Implementation not finished')
}

const layer0WeightFragmentShader = /* glsl */ `#version 300 es

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
    float x = texelFetch(xTex, fragCoord % 32, 0).r;
    float weight = texelFetch(weightTex, weightCoord, 0).r;
    y = x * weight;
  }
`

const layer0BiasLayer1FragmentShader = /* glsl */ `#version 300 es

  precision highp float;

  uniform sampler2D xTex;
  uniform sampler2D biasTex;

  out float y;

  void main() {
    ivec2 fragCoord = ivec2(gl_FragCoord);
    float x = texelFetch(xTex, fragCoord, 5).r * 1024.0;
    float bias = texelFetch(biasTex, fragCoord, 0).r;
    y = max(0.0, x + bias); // ReLU
  }
`

const layer2WeightFragmentShader = /* glsl */ `#version 300 es

  precision highp float;
  
  uniform sampler2D xTex;
  uniform sampler2D weightTex;

  out float y;

  void main() {
    ivec2 fragCoord = ivec2(gl_FragCoord);
    // TODO Rearrange the weights before uploading to texture
    ivec2 weightCoord = ivec2(
      (fragCoord.x % 32) + (fragCoord.y % 32) * 23,
      (fragCoord.x / 32) + (fragCoord.y / 32) * 23
    );
    float x = texelFetch(xTex, fragCoord % 32, 0).r;
    float weight = texelFetch(weightTex, weightCoord, 0).r;
    y = x * weight;
  }
`

const layer2BiasLayer3FragmentShader = /* glsl */ `#version 300 es

  precision highp float;

  uniform sampler2D xTex;
  uniform sampler2D biasTex;

  out float y;

  void main() {
    ivec2 fragCoord = ivec2(gl_FragCoord);
    float x = texelFetch(xTex, fragCoord, 5).r * 1024.0;
    float bias = texelFetch(biasTex, fragCoord, 0).r;
    y = max(0.0, x + bias); // ReLU
  }
`

const layer0WeightProgramInfo = twgl.createProgramInfo(gl, [
  vertexShader,
  layer0WeightFragmentShader,
])

const layer0BiasLayer1ProgramInfo = twgl.createProgramInfo(gl, [
  vertexShader,
  layer0BiasLayer1FragmentShader,
])

const layer2WeightProgramInfo = twgl.createProgramInfo(gl, [
  vertexShader,
  layer2WeightFragmentShader,
])

const layer2BiasLayer3ProgramInfo = twgl.createProgramInfo(gl, [
  vertexShader,
  layer2BiasLayer3FragmentShader,
])

// 32 = lowest power of 2 that is greater than or equal to 28
// 23 = ceil(sqrt(512))
// 529 = 23 * 23
// 736 = 32 * 23

const layer0Bias = new Float32Array(529)
layer0Bias.set(bias0Raw)

const layer2Bias = new Float32Array(529)
layer2Bias.set(biasRaw2)

type Textures = {
  x: WebGLTexture
  layer0Weight: WebGLTexture
  layer0Bias: WebGLTexture
  layer2Weight: WebGLTexture
  layer2Bias: WebGLTexture
}

const textures = await new Promise<Textures>((resolve) => {
  const result = twgl.createTextures(
    gl,
    {
      // TODO Consider grouping 4 successive R components into single RGBA pixel
      x: { src: 'data/fashion-mnist/0.png' },
      // TODO Set max mipmap level
      layer0Weight: {
        src: weight0,
        internalFormat: gl.R32F,
        width: 784,
        height: 512,
      },
      layer0Bias: {
        src: layer0Bias,
        internalFormat: gl.R32F,
        width: 23,
        height: 23,
      },
      layer2Weight: {
        src: weight2,
        internalFormat: gl.R32F,
        width: 512,
        height: 512,
      },
      layer2Bias: {
        src: layer2Bias,
        internalFormat: gl.R32F,
        width: 23,
        height: 23,
      },
    },
    () => resolve(result),
  ) as Textures
})

const layer0WeightFrameBufferInfo = twgl.createFramebufferInfo(
  gl,
  [{ internalFormat: gl.R32F }],
  736,
  736,
)

const layer0BiasLayer1FrameBufferInfo = twgl.createFramebufferInfo(
  gl,
  [{ internalFormat: gl.R32F }],
  23,
  23,
)

const layer2WeightFrameBufferInfo = twgl.createFramebufferInfo(
  gl,
  [{ internalFormat: gl.R32F }],
  736,
  736,
)

const layer2BiasLayer3FrameBufferInfo = twgl.createFramebufferInfo(
  gl,
  [{ internalFormat: gl.R32F }],
  23,
  23,
)

gl.viewport(0, 0, 736, 736)

const layer0WeightUniforms = {
  xTex: textures.x,
  weightTex: textures.layer0Weight,
}

gl.bindFramebuffer(gl.FRAMEBUFFER, layer0WeightFrameBufferInfo.framebuffer)
gl.useProgram(layer0WeightProgramInfo.program)
twgl.setBuffersAndAttributes(gl, layer0WeightProgramInfo, bufferInfo)
twgl.setUniforms(layer0WeightProgramInfo, layer0WeightUniforms)
twgl.drawBufferInfo(gl, bufferInfo)

gl.bindTexture(gl.TEXTURE_2D, layer0WeightFrameBufferInfo.attachments[0])
gl.generateMipmap(gl.TEXTURE_2D)

gl.viewport(0, 0, 23, 23)

const layer0BiasLayer1Uniforms = {
  xTex: layer0WeightFrameBufferInfo.attachments[0],
  biasTex: textures.layer0Bias,
}

gl.bindFramebuffer(gl.FRAMEBUFFER, layer0BiasLayer1FrameBufferInfo.framebuffer)
gl.useProgram(layer0BiasLayer1ProgramInfo.program)
twgl.setBuffersAndAttributes(gl, layer0BiasLayer1ProgramInfo, bufferInfo)
twgl.setUniforms(layer0BiasLayer1ProgramInfo, layer0BiasLayer1Uniforms)
twgl.drawBufferInfo(gl, bufferInfo)

// FIXME Error on Firefox
const hidden1 = new Float32Array(512)
gl.readPixels(0, 0, 23, 22, gl.RED, gl.FLOAT, hidden1)
gl.readPixels(0, 22, 6, 1, gl.RED, gl.FLOAT, hidden1, 506)
console.log(
  `Hidden 1: ${hidden1.slice(0, 10).join(', ')}, ... , ${hidden1
    .slice(-10)
    .join(', ')}`,
)

gl.viewport(0, 0, 736, 736)

const layer2WeightUniforms = {
  xTex: layer0BiasLayer1FrameBufferInfo.attachments[0],
  weightTex: textures.layer2Weight,
}

gl.bindFramebuffer(gl.FRAMEBUFFER, layer2WeightFrameBufferInfo.framebuffer)
gl.useProgram(layer2WeightProgramInfo.program)
twgl.setBuffersAndAttributes(gl, layer2WeightProgramInfo, bufferInfo)
twgl.setUniforms(layer2WeightProgramInfo, layer2WeightUniforms)
twgl.drawBufferInfo(gl, bufferInfo)

gl.bindTexture(gl.TEXTURE_2D, layer2WeightFrameBufferInfo.attachments[0])
gl.generateMipmap(gl.TEXTURE_2D)

gl.viewport(0, 0, 23, 23)

const layer2BiasLayer3Uniforms = {
  xTex: layer2WeightFrameBufferInfo.attachments[0],
  biasTex: textures.layer2Bias,
}

gl.bindFramebuffer(gl.FRAMEBUFFER, layer2BiasLayer3FrameBufferInfo.framebuffer)
gl.useProgram(layer2BiasLayer3ProgramInfo.program)
twgl.setBuffersAndAttributes(gl, layer2BiasLayer3ProgramInfo, bufferInfo)
twgl.setUniforms(layer2BiasLayer3ProgramInfo, layer2BiasLayer3Uniforms)
twgl.drawBufferInfo(gl, bufferInfo)

const hidden3 = new Float32Array(512)
gl.readPixels(0, 0, 23, 22, gl.RED, gl.FLOAT, hidden3)
gl.readPixels(0, 22, 6, 1, gl.RED, gl.FLOAT, hidden3, 506)
console.log(
  `Hidden 3: ${hidden3.slice(0, 10).join(', ')}, ... , ${hidden3
    .slice(-10)
    .join(', ')}`,
)
