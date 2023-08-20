import { VecMatMulResult } from './types'

export const setupVecMatMulCPU = (x: Float32Array, a: Float32Array) => {
  const y = new Float32Array(a.length / x.length)

  const vecMatMulCPU = async (): Promise<VecMatMulResult> => {
    const start = performance.now()
    for (let row = 0; row < y.length; row++) {
      y[row] = 0
      for (let i = 0; i < x.length; i++) {
        y[row] += a[row * x.length + i] * x[i]
      }
    }
    const time = performance.now() - start
    return { result: y, time }
  }

  return vecMatMulCPU
}
