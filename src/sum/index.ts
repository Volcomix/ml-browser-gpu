import '../benchmark.css'

import { setupSumCPU } from './cpu'
import { setupSumWebGL } from './webgl'
import {
  adapterInfo,
  setupSumWebGPUAtomic,
  setupSumWebGPURecursive,
  setupSumWebGPUSubgroup,
  setupSumWebGPUSubgroupTile,
  setupSumWebGPUTile,
  setupSumWebGPUVector,
} from './webgpu'

const setups = [
  setupSumCPU,
  setupSumWebGL,
  setupSumWebGPUAtomic,
  setupSumWebGPUTile,
  setupSumWebGPUVector,
  setupSumWebGPURecursive,
  setupSumWebGPUSubgroup,
  setupSumWebGPUSubgroupTile,
]

const generateInput = (count: number) => {
  const start = performance.now()
  const input = new Uint32Array(count).fill(1)
  const time = performance.now() - start
  console.log(`${generateInput.name}: ${time} ms`)
  return input
}

type RunLimitType = 'count' | 'duration'

const searchParams = new URLSearchParams(location.search)

const params = {
  minIntCount: Number(searchParams.get('minIntCount') ?? 2 ** 17),
  maxIntCount: Number(searchParams.get('maxIntCount') ?? 2 ** 25),
  runLimit: Number(searchParams.get('runLimit') ?? 500),
  runLimitType:
    searchParams.get('runLimitType') ?? ('duration' as RunLimitType),
}

type ParamName = keyof typeof params

const superscripts: Record<string, string> = {
  '0': '⁰',
  '1': '¹',
  '2': '²',
  '3': '³',
  '4': '⁴',
  '5': '⁵',
  '6': '⁶',
  '7': '⁷',
  '8': '⁸',
  '9': '⁹',
}

const formatIntCount = (count: number) => {
  return `${count} (2${[...String(Math.log2(count))]
    .map((value) => superscripts[value])
    .join('')})`
}

const populateIntCountOptions = (selectElement: HTMLSelectElement) => {
  for (let i = 2; i <= 25; i++) {
    const count = 2 ** i
    const option = document.createElement('option')
    option.value = String(count)
    option.textContent = formatIntCount(count)
    selectElement.appendChild(option)
  }
}

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
  const { minIntCount, maxIntCount } = params

  const intCountHeader = document.querySelector<HTMLTableCellElement>(
    'thead tr:first-of-type th:last-of-type',
  )!
  intCountHeader.colSpan = Math.log2(maxIntCount) - Math.log2(minIntCount) + 1

  const intCountValues = document.querySelector<HTMLTableRowElement>(
    'thead tr:last-of-type',
  )!
  const intCountValuesChildren: HTMLTableCellElement[] = []
  intCountValuesChildren.push(document.createElement('th'))
  for (let count = minIntCount; count <= maxIntCount; count *= 2) {
    const headerCell = document.createElement('th')
    headerCell.textContent = formatIntCount(count)
    intCountValuesChildren.push(headerCell)
  }
  intCountValues.replaceChildren(...intCountValuesChildren)

  const tableBody = document.querySelector('tbody')!
  const tableRows: HTMLTableRowElement[] = []
  for (const setupSum of setups) {
    const sumName = setupSum.name.replace('setupS', 's')

    const row = document.createElement('tr')

    const dataCell = document.createElement('td')
    dataCell.textContent = sumName
    row.appendChild(dataCell)

    for (let count = minIntCount; count <= maxIntCount; count *= 2) {
      const dataCell = document.createElement('td')
      dataCell.id = `${sumName}-${count}`
      dataCell.textContent = 'ready'
      row.appendChild(dataCell)
    }
    tableRows.push(row)
  }
  tableBody.replaceChildren(...tableRows)
}

initTable()

const minIntCountElement =
  document.querySelector<HTMLSelectElement>('[name=minIntCount]')!
populateIntCountOptions(minIntCountElement)
bindElement(minIntCountElement, 'minIntCount', initTable)

const maxIntCountElement =
  document.querySelector<HTMLSelectElement>('[name=maxIntCount]')!
populateIntCountOptions(maxIntCountElement)
bindElement(maxIntCountElement, 'maxIntCount', initTable)

bindElement(
  document.querySelector<HTMLInputElement>('[name=runLimit]')!,
  'runLimit',
)
bindElement(
  document.querySelector<HTMLSelectElement>('[name=runLimitType')!,
  'runLimitType',
)

document.querySelector('button')!.onclick = async () => {
  document.querySelectorAll('td[id]').forEach((cell) => {
    cell.className = ''
    cell.textContent = 'pending...'
  })
  await waitForUIUpdate()

  const { minIntCount, maxIntCount, runLimit, runLimitType } = params
  for (let count = minIntCount; count <= maxIntCount; count *= 2) {
    console.groupCollapsed(`${formatIntCount(count)} ints`)

    let firstResult: number | undefined
    const input = generateInput(count)
    const means: Record<string, number> = {}
    for (const setupSum of setups) {
      const sum = setupSum(input)

      const id = `${sum.name}-${count}`

      const cell = document.getElementById(id)!
      cell.textContent = 'running...'
      await waitForUIUpdate()

      console.groupCollapsed(sum.name)
      let latestResult: number | undefined
      let timeSum = 0
      let runCount = 0

      // Warm-up
      await sum()

      if (runLimitType === 'count') {
        for (let i = 0; i < runLimit; i++) {
          const { result, time } = await sum()
          console.log(`time: ${time} ms`)
          latestResult = result
          timeSum += time
          runCount++
        }
      } else {
        const start = performance.now()
        while (performance.now() - start < runLimit) {
          const { result, time } = await sum()
          console.log(`time: ${time} ms`)
          latestResult = result
          timeSum += time
          runCount++
        }
      }
      console.log(`result: ${latestResult}`)
      console.groupEnd()

      if (firstResult === undefined) {
        firstResult = latestResult
      }
      if (latestResult === firstResult) {
        means[`${sum.name}-${count}`] = timeSum / runCount
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
