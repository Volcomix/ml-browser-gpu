import '../benchmark.css'

import { setupVecMatMulCPU } from './cpu'
import {
  adapterInfo,
  setupVecMatMulWebGPUGlobMemCoalesce,
  setupVecMatMulWebGPUSimple,
} from './webgpu'

const sizes = ['784x512']
const setups = [
  setupVecMatMulCPU,
  setupVecMatMulWebGPUSimple,
  setupVecMatMulWebGPUGlobMemCoalesce,
]

const loadX = async () => {
  const response = await fetch(`/fashion-mnist/data/0.png`)
  const blob = await response.blob()
  const imageBitmap = await createImageBitmap(blob)
  const canvas = new OffscreenCanvas(28, 28)
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(imageBitmap, 0, 0)
  const imageData = ctx.getImageData(0, 0, 28, 28).data
  const x = new Float32Array(28 * 28)
  for (let i = 0; i < x.length; i++) {
    x[i] = imageData[i * 4] / 255
  }
  return x
}

const loadA = async () => {
  const response = await fetch(`/fashion-mnist/models/0-weight.gz`)
  const buffer = await response.arrayBuffer()
  return new Float32Array(buffer)
}

type RunLimitType = 'count' | 'duration'

const searchParams = new URLSearchParams(location.search)

const params = {
  runLimit: Number(searchParams.get('runLimit') ?? 500),
  runLimitType:
    searchParams.get('runLimitType') ?? ('duration' as RunLimitType),
}

type ParamName = keyof typeof params

const bindElement = (
  element: HTMLInputElement | HTMLSelectElement,
  paramName: ParamName,
  callback?: () => void,
) => {
  const value = params[paramName]
  element.value = typeof value === 'string' ? value : String(value)
  element.onchange = () => {
    ;(params[paramName] as typeof value) =
      typeof value === 'string' ? element.value : Number(element.value)
    const searchParams = new URLSearchParams(location.search)
    searchParams.set(paramName, element.value)
    history.replaceState(null, '', `?${searchParams}`)
    callback?.()
  }
}

const waitForUIUpdate = () => new Promise((resolve) => setTimeout(resolve, 10))

const updateGpuDetails = async () => {
  document.querySelector('.gpu__vendor span')!.textContent = adapterInfo.vendor
  document.querySelector('.gpu__architecture span')!.textContent =
    adapterInfo.architecture
}

updateGpuDetails()

const initTable = () => {
  const sizeRow = document.querySelector<HTMLTableRowElement>('thead tr')!
  const sizeRowChildren: HTMLTableCellElement[] = []
  sizeRowChildren.push(document.createElement('th'))
  for (const size of sizes) {
    const headerCell = document.createElement('th')
    headerCell.textContent = size
    sizeRowChildren.push(headerCell)
  }
  sizeRow.replaceChildren(...sizeRowChildren)

  const tableBody = document.querySelector('tbody')!
  const tableRows: HTMLTableRowElement[] = []
  for (const setup of setups) {
    const functionName = setup.name.replace(/setup([A-Z])/, (_, c: string) =>
      c.toLowerCase(),
    )

    const row = document.createElement('tr')

    const dataCell = document.createElement('td')
    dataCell.textContent = functionName
    row.appendChild(dataCell)

    for (const size of sizes) {
      const dataCell = document.createElement('td')
      dataCell.id = `${functionName}-${size}`
      dataCell.textContent = 'ready'
      row.appendChild(dataCell)
    }
    tableRows.push(row)
  }
  tableBody.replaceChildren(...tableRows)
}

initTable()

bindElement(
  document.querySelector<HTMLInputElement>('[name=runLimit]')!,
  'runLimit',
)
bindElement(
  document.querySelector<HTMLSelectElement>('[name=runLimitType')!,
  'runLimitType',
)

const [x, a] = await Promise.all([loadX(), loadA()])

document.querySelector('button')!.onclick = async () => {
  document.querySelectorAll('td[id]').forEach((cell) => {
    cell.className = ''
    cell.textContent = 'pending...'
  })
  await waitForUIUpdate()

  const { runLimit, runLimitType } = params
  for (const size of sizes) {
    console.groupCollapsed(size)

    let firstResult: Float32Array | undefined
    const means: Record<string, number> = {}
    for (const setup of setups) {
      const func = setup(x, a)

      const id = `${func.name}-${size}`

      const cell = document.getElementById(id)!
      cell.textContent = 'running...'
      await waitForUIUpdate()

      console.groupCollapsed(func.name)
      let latestResult: Float32Array | undefined
      let timeSum = 0
      let runCount = 0

      // Warm-up
      await func()

      if (runLimitType === 'count') {
        for (let i = 0; i < runLimit; i++) {
          const { result, time } = await func()
          console.log(`time: ${time} ms`)
          latestResult = result
          timeSum += time
          runCount++
        }
      } else {
        const start = performance.now()
        while (performance.now() - start < runLimit) {
          const { result, time } = await func()
          console.log(`time: ${time} ms`)
          latestResult = result
          timeSum += time
          runCount++
        }
      }
      console.log('result:', latestResult)
      console.groupEnd()

      if (firstResult === undefined) {
        firstResult = latestResult
      }
      if (
        latestResult?.every((value, index) => firstResult?.[index] === value)
      ) {
        means[`${func.name}-${size}`] = timeSum / runCount
        cell.textContent = 'completed'
      } else {
        cell.textContent = 'ERROR'
        cell.classList.add('result__cell--error')
      }
      await waitForUIUpdate()
    }
    console.groupEnd()

    let fastest = ''
    let fastestTime = Infinity
    let slowest = ''
    let slowestTime = -Infinity
    for (const [id, mean] of Object.entries(means)) {
      if (mean === undefined) {
        continue
      }
      if (mean < fastestTime) {
        fastest = id
        fastestTime = mean
      }
      if (mean > slowestTime) {
        slowest = id
        slowestTime = mean
      }
      document.getElementById(id)!.textContent = `${mean.toLocaleString(
        'en-US',
        { maximumFractionDigits: 3 },
      )} ms`
    }
    if (fastest) {
      document.getElementById(fastest)!.classList.add('result__cell--fastest')
    }
    if (slowest) {
      document.getElementById(slowest)!.classList.add('result__cell--slowest')
    }
    await waitForUIUpdate()
  }
}
