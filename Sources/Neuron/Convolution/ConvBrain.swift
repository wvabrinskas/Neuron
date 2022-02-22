//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/19/22.
//

import Foundation
import Logger


public class ConvBrain: Logger {
  public var logLevel: LogLevel = .low
  
  private var inputSize: (Int, Int, Int)
  private var fullyConnected: Brain
  private var learningRate: Float

  public init(learningRate: Float,
              inputSize: (Int, Int, Int),
              fullyConnected: Brain) {
    self.learningRate = learningRate
    self.inputSize = inputSize
    self.fullyConnected = fullyConnected
  }
  
  public func addFullyConnected(_ brain: Brain) {
    self.fullyConnected = brain
    if brain.compiled == false {
      brain.compile()
    }
  }

//  public func add(_ model: LobeDefinition) {
//    var lobe = Lobe(model: model,
//                    learningRate: learningRate)
//
//    if let bnModel = model as? NormalizedLobeModel  {
//      lobe = NormalizedLobe(model: bnModel,
//                            learningRate: learningRate)
//    }
//
//    if let convModel = model as? ConvolutionalLobeModel {
//      lobe = ConvolutionalLobe(model: convModel,
//                               learningRate: learningRate,
//                               optimizer: optimizer,
//                               initializer: initializer)
//    }
//
//    if let poolModel = model as? PoolingLobeModel {
//      lobe = PoolingLobe(model: poolModel,
//                         learningRate: learningRate)
//    }
//
//    self.lobes.append(lobe)
//  }
//
}
