//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/26/22.
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
  case realImageLoss = "Real Image Loss"
  case fakeImageLoss = "Fake Image Loss"
  case valAccuracy = "Validation Accuracy"
}

public protocol MetricLogger: AnyObject {
  var metricsToGather: Set<Metric> { get set }
  var metrics: [Metric: Float] { get set }
  var lock: NSLock { get }
  func addMetric(value: Float, key: Metric)
}

public extension MetricLogger {
  func addMetric(value: Float, key: Metric) {
    if metricsToGather.contains(key) {
      lock.with {
        metrics[key] = value
      }
    }
  }
}

internal protocol MetricCalculator: MetricLogger {
  var totalCorrectGuesses: Int { get set }
  var totalGuesses: Int { get set }
  var totalValCorrectGuesses: Int { get set }
  var totalValGuesses: Int { get set }
  
  func calculateValAccuracy(_ guess: [Float], label: [Float], binary: Bool) -> Float
  func calculateAccuracy(_ guess: [Float], label: [Float], binary: Bool) -> Float
}

internal extension MetricCalculator {
  func calculateValAccuracy(_ guess: [Float], label: [Float], binary: Bool) -> Float {
    //only useful for classification problems
    let max = label.indexOfMax
    let guessMax = guess.indexOfMax
    if binary {
      if max.1 - guessMax.1 < 0.5 {
        totalValCorrectGuesses += 1
      }
    } else {
      if max.0 == guessMax.0 {
        totalValCorrectGuesses += 1
      }
    }
    
    totalValGuesses += 1
    
    let accuracy = Float(totalValCorrectGuesses) / Float(totalValGuesses) * 100.0
    return accuracy
  }
  
  func calculateAccuracy(_ guess: [Float], label: [Float], binary: Bool = false) -> Float {
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
    return accuracy
  }

}

public class MetricsReporter: MetricCalculator {
  public var lock: NSLock = NSLock()
  internal var totalValCorrectGuesses: Int = 0
  internal var totalValGuesses: Int = 0
  
  internal var totalCorrectGuesses: Int = 0
  internal var totalGuesses: Int = 0
  
  private var frequency: Int
  private var currentStep: Int = 0
  
  public var metricsToGather: Set<Metric>
  public var metrics: [Metric : Float] = [:]
  public var receive: ((_ metrics: [Metric: Float]) -> ())? = nil
  
  public init(frequency: Int = 5, metricsToGather: Set<Metric>) {
    self.frequency = frequency
    self.metricsToGather = metricsToGather
  }
  
  internal func update(metric: Metric, value: Float) {
    addMetric(value: value, key: metric)
  }
  
  internal func report() {
    currentStep += 1
    if currentStep % frequency == 0 {
      receive?(metrics)
      currentStep = 0
    }
  }
}
