//
//  WelfordVariance.swift
//  Neuron
//
//  Created by William Vabrinskas on 7/16/25.
//

import Foundation
import NumSwift

public final class WelfordVariance {
  /// Per-depth-slice means stored as flat arrays
  @Atomic public private(set) var means: [ContiguousArray<Tensor.Scalar>] = []
  /// Per-depth-slice M2 accumulators stored as flat arrays
  @Atomic public private(set) var m2s: [ContiguousArray<Tensor.Scalar>] = []
  @Atomic public private(set) var iterations: Int = 0
  
  private var inputSize: TensorSize = .init(array: [])
  
  init(inputSize: TensorSize = .init(array: [])) {
    setInputSize(inputSize)
  }
  
  public func setInputSize(_ inputSize: TensorSize) {
    self.inputSize = inputSize
    reset()
  }
  
  public func update(_ inputs: Tensor) {
    iterations += 1
    let iterScalar = iterations.asTensorScalar
    
    for i in 0..<inputs.depthSliceCount {
      let inputSlice = inputs.depthSlice(i)
      let delta = NumSwiftFlat.subtract(inputSlice, means[i])
      means[i] = NumSwiftFlat.add(means[i], NumSwiftFlat.divide(delta, scalar: iterScalar))
      
      let delta2 = NumSwiftFlat.subtract(inputSlice, means[i])
      m2s[i] = NumSwiftFlat.add(m2s[i], NumSwiftFlat.multiply(delta, delta2))
    }
  }
  
  public func reset() {
    iterations = 0
    let sliceSize = inputSize.rows * inputSize.columns
    means = [ContiguousArray<Tensor.Scalar>](repeating: ContiguousArray(repeating: 0, count: sliceSize), count: inputSize.depth)
    m2s = [ContiguousArray<Tensor.Scalar>](repeating: ContiguousArray(repeating: 0, count: sliceSize), count: inputSize.depth)
  }
  
}
