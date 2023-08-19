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
  const response = await fetch(`fashion-mnist/models/${name}.gz`)
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
      { src: 'fashion-mnist/data/0.png' },
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

type Activation = (x: string) => string

const Identity: Activation = (x: string) => x
const ReLU: Activation = (x: string) => `max(vec4(0), ${x})`

type SetupMultiplySize = {
  source: Size
  tile: Size
}

const setupMultiply = (
  size: SetupMultiplySize,
  sourceComponent: 'R' | 'RGBA' = 'RGBA',
) => {
  const xComponent = sourceComponent === 'R' ? '.r' : '[fragCoord.x % 4]'

  const xCoord =
    size.source[0] === 1
      ? [0, `fragCoord.x + (fragCoord.y % ${size.tile[1]}) * ${size.tile[0]}`]
      : ['fragCoord.x', `fragCoord.y % ${size.tile[1]}`]

  if (sourceComponent === 'RGBA') {
    xCoord[1] = `(${xCoord[1]}) / 4`
  }

  const fragmentShader = /* glsl */ `#version 300 es

    precision highp float;

    uniform sampler2D xTex;
    uniform sampler2D wTex;

    out vec4 y;

    void main() {
      ivec2 fragCoord = ivec2(gl_FragCoord);
      ivec2 xCoord = ivec2(${xCoord.join(', ')});
      float x = texelFetch(xTex, xCoord, 0)${xComponent};
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

type SetupSumSize = {
  tile: Size
}

const setupSum = (size: SetupSumSize, activation: Activation = Identity) => {
  const tileCount = size.tile[0] / size.tile[1]
  const lod = Math.log2(size.tile[1])

  const x = Array.from(
    { length: tileCount },
    (_v, k) => `texelFetch(xTex, ivec2(${k}, fragCoord.y), ${lod})`,
  ).join(' + ')

  const fragmentShader = /* glsl */ `#version 300 es

    precision highp float;

    uniform sampler2D xTex;
    uniform sampler2D bTex;

    out vec4 y;

    void main() {
      ivec2 fragCoord = ivec2(gl_FragCoord);
      vec4 x = ${tileCount > 1 ? `(${x})` : x} * ${size.tile[1] ** 2}.0;
      vec4 b = texelFetch(bTex, fragCoord, 0);
      y = ${activation('x + b')};
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
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAX_LEVEL, lod)
    gl.generateMipmap(gl.TEXTURE_2D)
    process(programInfo, fbi, { xTex, bTex }, viewportWidth, viewportHeight)
  }

  return sum
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

const multiply0 = setupMultiply({ source: [32, 32], tile: [32, 32] }, 'R')
const sum0_1 = setupSum({ tile: [32, 32] }, ReLU)
const multiply2_4 = setupMultiply({ source: [1, 128], tile: [32, 16] })
const sum2_3 = setupSum({ tile: [32, 16] }, ReLU)
const sum4 = setupSum({ tile: [32, 16] })

const fbi = {
  '32x4096': createFramebufferInfo({ internalFormat: gl.RGBA32F }, 32, 4096),
  '1x128': createFramebufferInfo({ internalFormat: gl.RGBA32F }, 1, 128),
}

const predict = () => {
  multiply0(input, weight0, fbi['32x4096'], 32, 4096)
  sum0_1(fbi['32x4096'].attachment, bias0, fbi['1x128'], 1, 128)
  multiply2_4(fbi['1x128'].attachment, weight2, fbi['32x4096'], 32, 2048)
  sum2_3(fbi['32x4096'].attachment, bias2, fbi['1x128'], 1, 128)
  multiply2_4(fbi['1x128'].attachment, weight4, fbi['32x4096'], 32, 48)
  sum4(fbi['32x4096'].attachment, bias4, fbi['1x128'], 1, 3)
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

for (let i = 0; i < 5; i++) {
  console.log(separator)

  const start = performance.now()

  predict()
  gl.readPixels(0, 0, 1, 3, gl.RGBA, gl.FLOAT, outputData)
  const predicted = classes[argmax(output)]

  const time = performance.now() - start

  console.log(`Output: ${format(output, 4)}`)
  console.log(`Predicted: ${predicted}`)
  console.log(`Time: ${time}ms`)
}
console.log(separator)
