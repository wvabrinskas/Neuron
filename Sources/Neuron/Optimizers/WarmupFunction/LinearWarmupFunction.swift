//
//  LinearWarmupFunction.swift
//  Neuron
//
//  Created by William Vabrinskas on 3/31/26.
//

import Foundation

public class LinearWarmupFunction: BaseWarmupFunction {
  public override func step() {
    let newLr = targetLearningRate * (Tensor.Scalar(globalSteps) / Tensor.Scalar(warmupSteps))
    setWarmedLearningRate(newLr)
    
    super.step()
  }
}
