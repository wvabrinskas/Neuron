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
  @Atomic public private(set) var means: [Tensor.Value] = []
  /// Per-depth-slice M2 accumulators stored as flat arrays
  @Atomic public private(set) var m2s: [Tensor.Value] = []
  @Atomic public private(set) var iterations: Int = 0
  
  private var inputSize: TensorSize = .init(array: [])
  
  init(inputSize: TensorSize = .init(array: [])) {
    setInputSize(inputSize)
  }
  
  /// Configures input shape and resets accumulated statistics.
  ///
  /// - Parameter inputSize: Expected tensor shape for subsequent updates.
  public func setInputSize(_ inputSize: TensorSize) {
    self.inputSize = inputSize
    reset()
  }
  
  /// Incorporates one tensor into running Welford mean/variance statistics.
  ///
  /// - Parameter inputs: Input tensor sample to accumulate.
  public func update(_ inputs: Tensor) {
    iterations += 1
    let iterScalar = iterations.asTensorScalar
    
    for i in 0..<inputs.size.depth {
      let inputSlice = inputs.depthSlice(i)
      let delta = inputSlice - means[i]
      means[i] = means[i] + delta / Tensor.Scalar(iterScalar)
      
      let delta2 = inputSlice - means[i]
      m2s[i] = m2s[i] + delta * delta2
    }
  }
  
  /// Resets iteration count and running statistics to zeros.
  public func reset() {
    iterations = 0
    let sliceSize = inputSize.rows * inputSize.columns
    means = [Tensor.Value](repeating: Tensor.Value(repeating: 0, count: sliceSize), count: inputSize.depth)
    m2s = [Tensor.Value](repeating: Tensor.Value(repeating: 0, count: sliceSize), count: inputSize.depth)
  }
  
}
