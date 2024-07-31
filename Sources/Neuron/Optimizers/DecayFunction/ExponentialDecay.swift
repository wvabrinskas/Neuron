//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/9/24.
//

import Foundation
import NumSwift
import Numerics

public final class ExponentialDecay<N: TensorNumeric>: BaseDecayFunction<N> {
  public override func step() {
    let exp: Tensor<N>.Scalar
    if staircase {
      exp = Tensor<N>.Scalar(Int(globalSteps) / Int(decaySteps))
    } else {
      exp = globalSteps / decaySteps
    }
    
    decayedLearningRate = originalLearningRate * Tensor<N>.Scalar.pow(decayRate, exp)
    
    super.step()
  }
}
