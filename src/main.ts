import * as twgl from 'twgl.js'

import './style.css'

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
  return new Float32Array(buffer)
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

type Activation = (x: string) => string

const Identity = (x: string) => x
const ReLU = (x: string) => `max(vec4(0), ${x})`

const setupMultiply1D = (tileWidth: number, tileHeight: number) => {
  const fragmentShader = /* glsl */ `#version 300 es

    precision highp float;

    uniform sampler2D xTex;
    uniform sampler2D wTex;

    out vec4 y;

    void main() {
      ivec2 fragCoord = ivec2(gl_FragCoord);
      ivec2 xCoord = ivec2(0, (fragCoord.x + (fragCoord.y % ${tileHeight}) * ${tileWidth}) / 4);
      float x = texelFetch(xTex, xCoord, 0)[fragCoord.x % 4];
      vec4 w = texelFetch(wTex, fragCoord, 0);
      y = x * w;
    }
  `

  const programInfo = twgl.createProgramInfo(gl, [vertexShader, fragmentShader])

  const multiply1D = (
    xTex: WebGLTexture,
    wTex: WebGLTexture,
    fbi: FramebufferInfo,
    viewportWidth: number,
    viewportHeight: number,
  ) => {
    process(programInfo, fbi, { xTex, wTex }, viewportWidth, viewportHeight)
  }

  return multiply1D
}

const setupMultiply2D = (tileHeight: number) => {
  const fragmentShader = /* glsl */ `#version 300 es

    precision highp float;

    uniform sampler2D xTex;
    uniform sampler2D wTex;

    out vec4 y;

    void main() {
      ivec2 fragCoord = ivec2(gl_FragCoord);
      ivec2 xCoord = ivec2(fragCoord.x, fragCoord.y % ${tileHeight});
      float x = texelFetch(xTex, xCoord, 0).r;
      vec4 w = texelFetch(wTex, fragCoord, 0);
      y = x * w;
    }
  `

  const programInfo = twgl.createProgramInfo(gl, [vertexShader, fragmentShader])

  const multiply2D = (
    xTex: WebGLTexture,
    wTex: WebGLTexture,
    fbi: FramebufferInfo,
    viewportWidth: number,
    viewportHeight: number,
  ) => {
    process(programInfo, fbi, { xTex, wTex }, viewportWidth, viewportHeight)
  }

  return multiply2D
}

const setupSum1D = (
  tileSize: number,
  lod: number,
  activation: Activation = Identity,
) => {
  const fragmentShader = /* glsl */ `#version 300 es

    precision highp float;

    uniform sampler2D xTex;
    uniform sampler2D bTex;

    out vec4 y;

    void main() {
      ivec2 fragCoord = ivec2(gl_FragCoord);
      vec4 x = texelFetch(xTex, fragCoord, ${lod}) * ${tileSize}.0;
      vec4 b = texelFetch(bTex, fragCoord, 0);
      y = ${activation('x + b')};
    }
  `

  const programInfo = twgl.createProgramInfo(gl, [vertexShader, fragmentShader])

  const sum1D = (
    xTex: WebGLTexture,
    bTex: WebGLTexture,
    fbi: FramebufferInfo,
    viewportWidth: number,
    viewportHeight: number,
  ) => {
    gl.bindTexture(gl.TEXTURE_2D, xTex)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAX_LEVEL, lod)
    gl.generateMipmap(gl.TEXTURE_2D)
    process(programInfo, fbi, { xTex, bTex }, viewportWidth, viewportHeight)
  }

  return sum1D
}

// TODO Improve contract
const setupSum2D = (
  tileSize: number,
  tileCount: number,
  lod: number,
  activation: Activation = Identity,
) => {
  const fragmentShader = /* glsl */ `#version 300 es

    precision highp float;

    uniform sampler2D xTex;
    uniform sampler2D bTex;

    out vec4 y;

    void main() {
      ivec2 fragCoord = ivec2(gl_FragCoord);
      vec4 x = (${Array.from(
        { length: tileCount },
        (_v, k) => `texelFetch(xTex, ivec2(${k}, fragCoord.y), ${lod})`,
      ).join(' + ')}) * ${tileSize / tileCount}.0;
      vec4 b = texelFetch(bTex, fragCoord, 0);
      y = ${activation('x + b')};
    }
  `

  const programInfo = twgl.createProgramInfo(gl, [vertexShader, fragmentShader])

  const sum2D = (
    xTex: WebGLTexture,
    bTex: WebGLTexture,
    fbi: FramebufferInfo,
    viewportWidth: number,
    viewportHeight: number,
  ) => {
    gl.bindTexture(gl.TEXTURE_2D, xTex)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAX_LEVEL, lod)
    gl.generateMipmap(gl.TEXTURE_2D)
    process(programInfo, fbi, { xTex, bTex }, viewportWidth, viewportHeight)
  }

  return sum2D
}

const argmax = (x: Float32Array) => {
  return [...x].reduce(
    (acc, value, index) => (value > acc.max ? { max: value, index } : acc),
    { max: -Infinity, index: -1 },
  ).index
}

const format = (data: Float32Array, maximumFractionDigits: number) => {
  return [...data]
    .map((value) => value.toLocaleString('en-US', { maximumFractionDigits }))
    .join(', ')
}

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

const multiply2D = setupMultiply2D(32)
const multiply1D = setupMultiply1D(32, 16)
const sum1DReLU = setupSum1D(1024, 5, ReLU)
const sum2DReLU = setupSum2D(512, 2, 4, ReLU)
const sum2D = setupSum2D(512, 2, 4)

const outputData = new Float32Array(12)
const output = outputData.subarray(0, 10)

const classes = [
  'T-shirt/top',
  'Trouser',
  'Pullover',
  'Dress',
  'Coat',
  'Sandal',
  'Shirt',
  'Sneaker',
  'Bag',
  'Ankle boot',
]

const separator = '-'.repeat(100)

const predict = () => {
  const start = performance.now()

  multiply2D(input, weight0, fbi['32x4096'], 32, 4096)
  sum1DReLU(fbi['32x4096'].attachment, bias0, fbi['1x128'], 1, 128)
  multiply1D(fbi['1x128'].attachment, weight2, fbi['32x4096'], 32, 2048)
  sum2DReLU(fbi['32x4096'].attachment, bias2, fbi['1x128'], 1, 128)
  multiply1D(fbi['1x128'].attachment, weight4, fbi['32x4096'], 32, 48)
  sum2D(fbi['32x4096'].attachment, bias4, fbi['1x128'], 1, 3)

  const gpuTime = performance.now() - start

  gl.readPixels(0, 0, 1, 3, gl.RGBA, gl.FLOAT, outputData)
  const predicted = classes[argmax(output)]

  const totalTime = performance.now() - start

  console.log(`Output: ${format(output, 4)}`)
  console.log(`Predicted: ${predicted}`)
  console.log(`GPU time: ${gpuTime}ms`)
  console.log(`Total time: ${totalTime}ms`)
}

for (let i = 0; i < 5; i++) {
  console.log(separator)
  predict()
}
console.log(separator)
