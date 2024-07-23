//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/9/24.
//

import Foundation
import NumSwift
import Numerics

public final class ExponentialDecay: BaseDecayFunction {
  public override func step() {
    let exp: Tensor.Scalar
    if staircase {
      exp = Tensor.Scalar(Int(globalSteps) / Int(decaySteps))
    } else {
      exp = globalSteps / decaySteps
    }
    
    decayedLearningRate = originalLearningRate * Tensor.Scalar.pow(decayRate, exp)
    
    super.step()
  }
}
