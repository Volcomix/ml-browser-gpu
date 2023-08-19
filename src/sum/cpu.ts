import { SumResult } from './types'

export const setupSumCPU = (input: Uint32Array) => {
  const sumCPU = (): SumResult => {
    const start = performance.now()
    const result = input.reduce((a, b) => a + b)
    const time = performance.now() - start
    return { result, time }
  }

  return sumCPU
}
