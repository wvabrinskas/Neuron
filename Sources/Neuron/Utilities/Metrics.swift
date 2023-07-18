//
//  File.swift
//
//
//  Created by William Vabrinskas on 5/26/22.
//

import Foundation
import NumSwift

public enum Metric: String {
  case loss
  case accuracy
  case valLoss
  case generatorLoss
  case criticLoss
  case gradientPenalty
  case realImageLoss
  case fakeImageLoss
  case valAccuracy
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

  func calculateAccuracy(_ guess: Tensor, label: Tensor, binary: Bool, running: Bool) -> Float
  func calculateValAccuracy(_ guess: Tensor, label: Tensor, binary: Bool,  running: Bool) -> Float

}

internal extension MetricCalculator {
  func calculateValAccuracy(_ guess: Tensor, label: Tensor, binary: Bool, running: Bool = false) -> Float {
    var totalCorrect = 0
    var totalGuess = 0
    
    typealias Max = (UInt, Float)
    
    func perform(max: Max, guessMax: Max) -> Int {
      if binary {
        if max.1 - guessMax.1 < 0.5 {
          return 1
        }
      } else {
        if max.0 == guessMax.0 {
          return 1
        }
      }
      return 0
    }
        
    for d in 0..<guess.value.count {
      for r in 0..<guess.value[d].count {
        let guessMax = guess.value[d][r].indexOfMax
        let labelMax = label.value[d][r].indexOfMax
        totalCorrect += perform(max: labelMax, guessMax: guessMax)
        totalValCorrectGuesses += perform(max: labelMax, guessMax: guessMax)
        totalGuess += 1
        totalValGuesses += 1
      }
    }
    
    let runningAccuracy = Float(totalValCorrectGuesses) / Float(totalValGuesses) * 100.0
    let accuracy = Float(totalCorrect) / Float(totalGuess) * 100.0
    return running ? runningAccuracy : accuracy
  }
  
  func calculateAccuracy(_ guess: Tensor, label: Tensor, binary: Bool, running: Bool = false) -> Float {
    
    var totalCorrect = 0
    var totalGuess = 0
    
    typealias Max = (UInt, Float)
    
    func perform(max: Max, guessMax: Max) -> Int {
      if binary {
        if max.1 - guessMax.1 < 0.5 {
          return 1
        }
      } else {
        if max.0 == guessMax.0 {
          return 1
        }
      }
      return 0
    }
        
    for d in 0..<guess.value.count {
      for r in 0..<guess.value[d].count {
        let guessMax = guess.value[d][r].indexOfMax
        let labelMax = label.value[d][r].indexOfMax
        totalCorrect += perform(max: labelMax, guessMax: guessMax)
        totalCorrectGuesses += perform(max: labelMax, guessMax: guessMax)
        totalGuess += 1
        totalGuesses += 1
      }
    }
    
    let runningAccuracy = Float(totalCorrectGuesses) / Float(totalGuesses) * 100.0
    let accuracy = Float(totalCorrect) / Float(totalGuess) * 100.0
    return running ? runningAccuracy : accuracy
  }
}

@dynamicMemberLookup
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
  
  public subscript(dynamicMember member: String) -> Float? {
    guard let metric = Metric(rawValue: member) else { return nil }
    return metrics[metric]
  }
  
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
