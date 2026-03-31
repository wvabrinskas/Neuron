//
//  LinearWarmupFunction.swift
//  Neuron
//
//  Created by William Vabrinskas on 3/31/26.
//

import Foundation

public class CosineWarmupFunction: BaseWarmupFunction {
  public override func step() {
    let newLr = targetLearningRate * 0.5 * (1 - Tensor.Scalar.cos(Tensor.Scalar.pi * Tensor.Scalar(globalSteps) / Tensor.Scalar(warmupSteps)))
    setWarmedLearningRate(newLr)
    
    super.step()
  }
}
