//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/13/22.
//

import Foundation
import Logger

internal protocol GANDefinition: MetricCalculator {
  var generator: Brain? { get set }
  var discriminator: Brain? { get set }
  var batchSize: Int { get set }
  var lossFunction: GANLossFunction { get set }
  var discriminatorNoiseFactor: Float? { get set }
  var epochs: Int { get set }
  var logLevel: LogLevel { get set }
  var randomNoise: () -> [Float] { get set }
  var validateGenerator: (_ output: [Float]) -> Bool { get set }
  var criticTrainPerEpoch: Int { get set }
  var discriminatorLossHistory: [Float] { get set }
  var generatorLossHistory: [Float] { get set }
  var gradientPenaltyHistory: [Float] { get set }
  
  func criticStep(real: [TrainingData],
                  fake: [TrainingData]) -> Float
  func generatorStep(fake: [TrainingData]) -> Float
}
