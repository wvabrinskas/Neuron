//
//  File.swift
//
//
//  Created by William Vabrinskas on 5/26/22.
//

import Foundation
import NumSwift

/// An enumeration of supported metric types used to track training and evaluation statistics.
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

/// A protocol that defines the interface for objects that collect and store training metrics.
public protocol MetricLogger: AnyObject {
  var metricsToGather: Set<Metric> { get set }
  var metrics: [Metric: Tensor.Scalar] { get set }
  var lock: NSLock { get }
  /// Records a metric value when the metric is enabled for gathering.
  ///
  /// - Parameters:
  ///   - value: Metric value to store.
  ///   - key: Metric key identifying the value.
  func addMetric(value: Tensor.Scalar, key: Metric)
}

public extension MetricLogger {
  /// Default metric-recording implementation guarded by a lock.
  ///
  /// - Parameters:
  ///   - value: Metric value to store.
  ///   - key: Metric key identifying the value.
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
        
    for d in 0..<guess.size.depth {
      let guessMax = guess.depthSlice(d).indexOfMax
      let labelMax = label.depthSlice(d).indexOfMax
      totalCorrect += perform(max: labelMax, guessMax: guessMax)
      totalValCorrectGuesses += perform(max: labelMax, guessMax: guessMax)
      totalGuess += 1
      totalValGuesses += 1
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
        
    for d in 0..<guess.size.depth {
      let guessMax = guess.depthSlice(d).indexOfMax
      let labelMax = label.depthSlice(d).indexOfMax
      totalCorrect += perform(max: labelMax, guessMax: guessMax)
      totalCorrectGuesses += perform(max: labelMax, guessMax: guessMax)
      totalGuess += 1
      totalGuesses += 1
    }
    
    let runningAccuracy = Tensor.Scalar(totalCorrectGuesses) / Tensor.Scalar(totalGuesses) * 100.0
    let accuracy = Tensor.Scalar(totalCorrect) / Tensor.Scalar(totalGuess) * 100.0
    return running ? runningAccuracy : accuracy
  }
}

@dynamicMemberLookup
/// A class that collects, aggregates, and periodically reports training metrics during model training loops.
public class MetricsReporter: MetricCalculator {
/// A lock used to synchronize concurrent access to shared metric state.
  public var lock: NSLock = NSLock()
  internal var totalValCorrectGuesses: Int = 0
  internal var totalValGuesses: Int = 0
  
  internal var totalCorrectGuesses: Int = 0
  internal var totalGuesses: Int = 0
  
  private var frequency: Int
  private var currentStep: Int = 0
  private var timers: [Metric: [CFAbsoluteTime]] = [:]
  
  private var timerQueue = SynchronousOperationQueue(name: "metrics_reporter")
  
/// The set of metrics that this reporter is configured to gather and record.
  public var metricsToGather: Set<Metric>
/// A dictionary storing the current scalar values for each recorded metric.
  public var metrics: [Metric : Tensor.Scalar] = [:]
/// An optional closure called with the current metrics dictionary each time the reporting frequency threshold is reached.
  public var receive: ((_ metrics: [Metric: Tensor.Scalar]) -> ())? = nil
  
  deinit {
    timerQueue.cancelAllOperations()
  }
  
/// Retrieves a metric value by its raw string name using dynamic member lookup.
  ///
  /// - Parameter member: The raw string name of the metric to look up.
  /// - Returns: The scalar value for the matching metric, or `nil` if the name does not correspond to a known metric.
  public subscript(dynamicMember member: String) -> Tensor.Scalar? {
    guard let metric = Metric(rawValue: member) else { return nil }
    return metrics[metric]
  }
  
  /// Creates a metrics reporter for optimizer/model training loops.
  ///
  /// - Parameters:
  ///   - frequency: Number of report cycles between `receive` callbacks.
  ///   - metricsToGather: Metric keys that should be recorded.
  public init(frequency: Int = 5, metricsToGather: Set<Metric>) {
    self.frequency = frequency
    self.metricsToGather = metricsToGather
  }
  
  /// Starts a timer sample for the specified metric.
  ///
  /// - Parameter metric: Timing metric key to begin.
  public func startTimer(metric: Metric) {
    timerQueue.addBarrierBlock { [weak self] in
      guard let self = self else { return }
      
      if var hasTimers = self.timers[metric] {
        hasTimers.append(CFAbsoluteTimeGetCurrent())
        self.timers[metric] = hasTimers
      } else {
        self.timers[metric] = [CFAbsoluteTimeGetCurrent()]
      }
    }
  }
  
  /// Ends a timer sample and records its average elapsed value.
  ///
  /// - Parameter metric: Timing metric key to finalize.
  public func endTimer(metric: Metric) {
    timerQueue.waitUntilAllOperationsAreFinished()
    
    timerQueue.addBarrierBlock { [weak self] in
      guard let self = self else { return }
      
      if let timer = self.timers[metric] {
        let result = timer.map { Tensor.Scalar(CFAbsoluteTimeGetCurrent() - $0) }
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
