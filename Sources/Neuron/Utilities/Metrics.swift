//
//  File.swift
//  
//
//  Created by William Vabrinskas on 3/15/22.
//

import Foundation
import NumSwift

public enum Metric: String {
  case loss = "Training Loss"
  case accuracy = "Accuracy"
  case valLoss = "Validation Loss"
  case generatorLoss = "Generator Loss"
  case criticLoss = "Critic Loss"
  case gradientPenalty = "Gradient Penalty"
}

public protocol MetricLogger: AnyObject {
  var metricsToGather: Set<Metric> { get set }
  var metrics: [Metric: Float] { get set }
  func addMetric(value: Float, key: Metric)
}

public extension MetricLogger {
  func addMetric(value: Float, key: Metric) {
    if metricsToGather.contains(key) {
      metrics[key] = value
    }
  }
}

internal protocol MetricCalculator: MetricLogger {
  var totalCorrectGuesses: Int { get set }
  var totalGuesses: Int { get set }
  
  func calculateAccuracy(_ guess: [Float], label: [Float], binary: Bool)
}

internal extension MetricCalculator {
  func calculateAccuracy(_ guess: [Float], label: [Float], binary: Bool = false) {
    //only useful for classification problems
    let max = label.indexOfMax
    let guessMax = guess.indexOfMax
    if binary {
      if max.1 - guessMax.1 < 0.5 {
        totalCorrectGuesses += 1
      }
    } else {
      if max.0 == guessMax.0 {
        totalCorrectGuesses += 1
      }
    }
    totalGuesses += 1
    
    let accuracy = Float(totalCorrectGuesses) / Float(totalGuesses) * 100.0
    addMetric(value: accuracy, key: .accuracy)
  }

}

