//
//  WelfordVariance.swift
//  Neuron
//
//  Created by William Vabrinskas on 7/16/25.
//

import Foundation
import NumSwift

public final class WelfordVariance {
  @Atomic public private(set) var means: [[[Tensor.Scalar]]] = []
  @Atomic public private(set) var m2s: [[[Tensor.Scalar]]] = []
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
    
    for i in 0..<inputs.value.count {
      let delta = inputs.value[i] - means[i]
      let delta2Means = means[i] + (delta / iterations.asTensorScalar)
      
      means[i] = delta2Means
      
      let delta2 = inputs.value[i] - delta2Means
      
      m2s[i] = m2s[i] + (delta * delta2)
    }
  }
  
  public func reset() {
    iterations = 0
    
    means = NumSwift.zerosLike((inputSize.rows, inputSize.columns, inputSize.depth))
    m2s = NumSwift.zerosLike((inputSize.rows, inputSize.columns, inputSize.depth))
  }
  
}
