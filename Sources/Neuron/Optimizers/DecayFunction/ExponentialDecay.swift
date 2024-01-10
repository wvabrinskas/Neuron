//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/9/24.
//

import Foundation
import NumSwift

public final class ExponentialDecay: BaseDecayFunction {
  public override func step() {
    let exp: Float
    if staircase {
      exp = Float(Int(globalSteps) / Int(decaySteps))
    } else {
      exp = globalSteps / decaySteps
    }
    
    decayedLearningRate = originalLearningRate * pow(decayRate, exp)
    
    super.step()
  }
}
