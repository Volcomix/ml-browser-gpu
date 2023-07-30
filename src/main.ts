import * as twgl from 'twgl.js'

import './style.css'

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

type FramebufferInfo = Omit<twgl.FramebufferInfo, 'attachments'> & {
  attachment: WebGLTexture
}

const createFramebufferInfo = (
  gl: WebGLRenderingContext,
  attachment: twgl.AttachmentOptions,
  width: number,
  height: number,
): FramebufferInfo => {
  const fbi = twgl.createFramebufferInfo(gl, [attachment], width, height)
  return {
    framebuffer: fbi.framebuffer,
    attachment: fbi.attachments[0],
    width: fbi.width,
    height: fbi.height,
  }
}

const process = (
  programInfo: twgl.ProgramInfo,
  fbi: FramebufferInfo,
  uniforms: Record<string, unknown>,
  viewportWidth: number,
  viewportHeight: number,
) => {
  gl.viewport(0, 0, viewportWidth, viewportHeight)

  gl.bindFramebuffer(gl.FRAMEBUFFER, fbi.framebuffer)
  gl.useProgram(programInfo.program)
  twgl.setBuffersAndAttributes(gl, programInfo, bufferInfo)
  twgl.setUniforms(programInfo, uniforms)
  twgl.drawBufferInfo(gl, bufferInfo)
}

const fetchParameter = async (name: string) => {
  const response = await fetch(`models/fashion-mnist/${name}.gz`)
  const buffer = await response.arrayBuffer()
  const result = new Float32Array(buffer)

  console.log(`${name}: ${[...result.slice(0, 3)].join(', ')}, ...`)

  return result
}

type Size = [number, number]

type WeightSize = {
  source: Size
  destination: Size
  tile: Size
  viewport?: Size
}

const loadWeight = async (parameterName: string, size: WeightSize) => {
  const tex = twgl.createTexture(gl, {
    src: await fetchParameter(parameterName),
    internalFormat: gl.R32F,
    width: size.source[0],
    height: size.source[1],
  })

  const fbi = createFramebufferInfo(
    gl,
    { internalFormat: gl.RGBA32F },
    size.destination[0],
    size.destination[1],
  )

  const fragmentShader = /* glsl */ `#version 300 es
  
    precision highp float;
  
    uniform sampler2D tex;
  
    out vec4 result;
  
    void main() {
      ivec2 fragCoord = ivec2(gl_FragCoord);
      int x = fragCoord.x + (fragCoord.y % ${size.tile[1]}) * ${size.tile[0]};
      int y  = 4 * (fragCoord.y / ${size.tile[1]});
      result = vec4(
        texelFetch(tex, ivec2(x, y), 0).r,
        texelFetch(tex, ivec2(x, y + 1), 0).r,
        texelFetch(tex, ivec2(x, y + 2), 0).r,
        texelFetch(tex, ivec2(x, y + 3), 0).r
      );
    }
  `

  const programInfo = twgl.createProgramInfo(gl, [vertexShader, fragmentShader])

  const viewport = size.viewport ?? size.destination
  process(programInfo, fbi, { tex }, viewport[0], viewport[1])

  gl.deleteTexture(tex)
  gl.deleteFramebuffer(fbi.framebuffer)
  gl.deleteProgram(programInfo.program)

  return fbi.attachment
}

const loadBias = async (parameterName: string, size: number) => {
  let data = await fetchParameter(parameterName)

  if (size * 4 !== data.length) {
    const newData = new Float32Array(size * 4)
    newData.set(data)
    data = newData
  }

  return twgl.createTexture(gl, {
    src: data,
    internalFormat: gl.RGBA32F,
    width: 1,
    height: size,
  })
}

const loadInput = () => {
  return new Promise<WebGLTexture>((resolve, reject) =>
    twgl.createTexture(
      gl,
      { src: 'data/fashion-mnist/0.png' },
      (err, texture) => {
        if (err) {
          reject(err)
        } else {
          resolve(texture)
        }
      },
    ),
  )
}

type FramebufferInfos = {
  '32x4096': FramebufferInfo
  '1x128': FramebufferInfo
}

const createFramebufferInfos = (): FramebufferInfos => {
  return {
    '32x4096': createFramebufferInfo(
      gl,
      { internalFormat: gl.RGBA32F },
      32,
      4096,
    ),
    '1x128': createFramebufferInfo(gl, { internalFormat: gl.RGBA32F }, 1, 128),
  }
}

const setupMultiply = () => {
  const fragmentShader = /* glsl */ `#version 300 es

    precision highp float;

    uniform sampler2D xTex;
    uniform sampler2D wTex;

    out vec4 y;

    void main() {
      ivec2 fragCoord = ivec2(gl_FragCoord);
      ivec2 xCoord = ivec2(fragCoord.x, fragCoord.y % 32);
      float x = texelFetch(xTex, xCoord, 0).r;
      vec4 w = texelFetch(wTex, fragCoord, 0);
      y = x * w;
    }
  `

  const programInfo = twgl.createProgramInfo(gl, [vertexShader, fragmentShader])

  const multiply = (
    xTex: WebGLTexture,
    wTex: WebGLTexture,
    fbi: FramebufferInfo,
    viewportWidth: number,
    viewportHeight: number,
  ) => {
    process(programInfo, fbi, { xTex, wTex }, viewportWidth, viewportHeight)
  }

  return multiply
}

const setupSum = () => {
  const fragmentShader = /* glsl */ `#version 300 es

    precision highp float;

    uniform sampler2D xTex;
    uniform sampler2D bTex;

    out vec4 y;

    void main() {
      ivec2 fragCoord = ivec2(gl_FragCoord);
      vec4 x = texelFetch(xTex, fragCoord, 5) * 1024.0;
      vec4 b = texelFetch(bTex, fragCoord, 0);
      y = max(vec4(0), x + b); // ReLU
    }
  `

  const programInfo = twgl.createProgramInfo(gl, [vertexShader, fragmentShader])

  const sum = (
    xTex: WebGLTexture,
    bTex: WebGLTexture,
    fbi: FramebufferInfo,
    viewportWidth: number,
    viewportHeight: number,
  ) => {
    gl.bindTexture(gl.TEXTURE_2D, xTex)
    // TODO Set the right TEXTURE_BASE_LEVEL and TEXTURE_MAX_LEVEL before calling generateMipmap
    gl.generateMipmap(gl.TEXTURE_2D)
    process(programInfo, fbi, { xTex, bTex }, viewportWidth, viewportHeight)
  }

  return sum
}

// TODO Remove linter ignore
// eslint-disable-next-line @typescript-eslint/no-unused-vars
const [weight0, bias0, weight2, bias2, weight4, bias4, input] =
  await Promise.all([
    loadWeight('0-weight', {
      source: [784, 512],
      destination: [32, 4096],
      tile: [28, 32],
      viewport: [28, 4092],
    }),
    loadBias('0-bias', 128),

    loadWeight('2-weight', {
      source: [512, 512],
      destination: [32, 2048],
      tile: [32, 16],
    }),
    loadBias('2-bias', 128),

    loadWeight('4-weight', {
      source: [512, 10],
      destination: [32, 64],
      tile: [32, 16],
      viewport: [32, 48],
    }),
    loadBias('4-bias', 4),

    loadInput(),
  ])

const fbi = createFramebufferInfos()

const multiply = setupMultiply()
const sum = setupSum()

multiply(input, weight0, fbi['32x4096'], 28, 4092)
sum(fbi['32x4096'].attachment, bias0, fbi['1x128'], 1, 128)

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
