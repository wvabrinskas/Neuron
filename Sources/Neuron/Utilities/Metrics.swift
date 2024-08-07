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
  case batchTime
  case optimizerRunTime
  case batchConcurrency
}

public protocol MetricLogger: AnyObject {
  var metricsToGather: Set<Metric> { get set }
  var metrics: [Metric: Tensor.Scalar] { get set }
  var lock: NSLock { get }
  func addMetric(value: Tensor.Scalar, key: Metric)
}

public extension MetricLogger {
  func addMetric(value: Tensor.Scalar, key: Metric) {
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

  func calculateAccuracy(_ guess: Tensor, label: Tensor, binary: Bool, running: Bool) -> Tensor.Scalar
  func calculateValAccuracy(_ guess: Tensor, label: Tensor, binary: Bool,  running: Bool) -> Tensor.Scalar
  func startTimer(metric: Metric)
  func endTimer(metric: Metric)
}

internal extension MetricCalculator {
  func calculateValAccuracy(_ guess: Tensor, label: Tensor, binary: Bool, running: Bool = false) -> Tensor.Scalar {
    var totalCorrect = 0
    var totalGuess = 0
    
    typealias Max = (UInt, Tensor.Scalar)
    
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
    
    let runningAccuracy = Tensor.Scalar(totalValCorrectGuesses) / Tensor.Scalar(totalValGuesses) * 100.0
    let accuracy = Tensor.Scalar(totalCorrect) / Tensor.Scalar(totalGuess) * 100.0
    return running ? runningAccuracy : accuracy
  }
  
  func calculateAccuracy(_ guess: Tensor, label: Tensor, binary: Bool, running: Bool = false) -> Tensor.Scalar {
    
    var totalCorrect = 0
    var totalGuess = 0
    
    typealias Max = (UInt, Tensor.Scalar)
    
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
    
    let runningAccuracy = Tensor.Scalar(totalCorrectGuesses) / Tensor.Scalar(totalGuesses) * 100.0
    let accuracy = Tensor.Scalar(totalCorrect) / Tensor.Scalar(totalGuess) * 100.0
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
  private var timers: [Metric: [Date]] = [:]
  
  private var timerQueue = SynchronousOperationQueue(name: "metrics_reporter")
  
  public var metricsToGather: Set<Metric>
  public var metrics: [Metric : Tensor.Scalar] = [:]
  public var receive: ((_ metrics: [Metric: Tensor.Scalar]) -> ())? = nil
  
  deinit {
    timerQueue.cancelAllOperations()
  }
  
  public subscript(dynamicMember member: String) -> Tensor.Scalar? {
    guard let metric = Metric(rawValue: member) else { return nil }
    return metrics[metric]
  }
  
  public init(frequency: Int = 5, metricsToGather: Set<Metric>) {
    self.frequency = frequency
    self.metricsToGather = metricsToGather
  }
  
  public func startTimer(metric: Metric) {
    timerQueue.addBarrierBlock { [weak self] in
      guard let self = self else { return }
      
      if var hasTimers = self.timers[metric] {
        hasTimers.append(Date())
        self.timers[metric] = hasTimers
      } else {
        self.timers[metric] = [Date()]
      }
    }
  }
  
  public func endTimer(metric: Metric) {
    timerQueue.waitUntilAllOperationsAreFinished()
    
    timerQueue.addBarrierBlock { [weak self] in
      guard let self = self else { return }
      
      if let timer = self.timers[metric] {
        let result = timer.map { Tensor.Scalar(Date().timeIntervalSince1970 - $0.timeIntervalSince1970) }
        let average = result.average
        self.addMetric(value: average,
                       key: metric)
        self.timers.removeValue(forKey: metric)
      }
    }
  }
  
  internal func update(metric: Metric, value: Tensor.Scalar) {
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
