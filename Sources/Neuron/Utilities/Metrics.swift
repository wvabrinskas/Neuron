//
//  File.swift
//
//
//  Created by William Vabrinskas on 5/26/22.
//

import Foundation
import NumSwift

/// Enumeration of available metrics for tracking training progress
/// Each metric represents a different aspect of model performance or training efficiency
public enum Metric: String {
  /// Training loss value
  case loss
  /// Training accuracy percentage
  case accuracy
  /// Validation loss value
  case valLoss
  /// Generator loss (for GANs)
  case generatorLoss
  /// Critic/Discriminator loss (for GANs)
  case criticLoss
  /// Gradient penalty (for WGAN-GP)
  case gradientPenalty
  /// Real image loss (for GANs)
  case realImageLoss
  /// Fake image loss (for GANs)
  case fakeImageLoss
  /// Validation accuracy percentage
  case valAccuracy
  /// Time taken for batch processing
  case batchTime
  /// Time taken for optimizer operations
  case optimizerRunTime
  /// Batch concurrency level
  case batchConcurrency
}

/// Protocol for objects that can log and track metrics
/// Provides thread-safe metric collection with selective gathering
public protocol MetricLogger: AnyObject {
  /// Set of metrics to actively collect
  var metricsToGather: Set<Metric> { get set }
  /// Dictionary storing current metric values
  var metrics: [Metric: Tensor.Scalar] { get set }
  /// Thread synchronization lock for safe concurrent access
  var lock: NSLock { get }
  
  /// Adds a metric value if it's in the collection set
  /// - Parameters:
  ///   - value: The metric value to store
  ///   - key: The metric type identifier
  func addMetric(value: Tensor.Scalar, key: Metric)
}

/// Default implementation of MetricLogger protocol
public extension MetricLogger {
  /// Thread-safe method to add a metric value
  /// Only adds the metric if it's in the metricsToGather set
  /// - Parameters:
  ///   - value: The metric value to store
  ///   - key: The metric type identifier
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

/// MetricsReporter class for comprehensive training metrics collection and reporting
/// Provides accuracy calculation, timing, and customizable metric reporting
/// Supports dynamic member lookup for convenient metric access
@dynamicMemberLookup
public class MetricsReporter: MetricCalculator {
  /// Thread synchronization lock for safe concurrent access
  public var lock: NSLock = NSLock()
  /// Total correct validation predictions (running total)
  internal var totalValCorrectGuesses: Int = 0
  /// Total validation predictions made (running total)
  internal var totalValGuesses: Int = 0
  /// Total correct training predictions (running total)
  internal var totalCorrectGuesses: Int = 0
  /// Total training predictions made (running total)
  internal var totalGuesses: Int = 0
  
  /// How often to report metrics (every N steps)
  private var frequency: Int
  /// Current step counter for frequency-based reporting
  private var currentStep: Int = 0
  /// Active timers for measuring operation durations
  private var timers: [Metric: [Date]] = [:]
  /// Serial queue for thread-safe timer operations
  private var timerQueue = SynchronousOperationQueue(name: "metrics_reporter")
  
  /// Set of metrics to actively collect
  public var metricsToGather: Set<Metric>
  /// Dictionary storing current metric values
  public var metrics: [Metric : Tensor.Scalar] = [:]
  /// Callback closure called when metrics are reported
  public var receive: ((_ metrics: [Metric: Tensor.Scalar]) -> ())? = nil
  
  /// Cleans up timer operations when the reporter is deallocated
  deinit {
    timerQueue.cancelAllOperations()
  }
  
  /// Dynamic member lookup for convenient metric access
  /// Allows accessing metrics using dot notation (e.g., reporter.loss)
  /// - Parameter member: The metric name as a string
  /// - Returns: The metric value if it exists, nil otherwise
  public subscript(dynamicMember member: String) -> Tensor.Scalar? {
    guard let metric = Metric(rawValue: member) else { return nil }
    return metrics[metric]
  }
  
  /// Initializes a MetricsReporter with specified frequency and metrics to track
  /// - Parameters:
  ///   - frequency: How often to report metrics (every N steps). Default: 5
  ///   - metricsToGather: Set of metrics to actively collect
  public init(frequency: Int = 5, metricsToGather: Set<Metric>) {
    self.frequency = frequency
    self.metricsToGather = metricsToGather
  }
  
  /// Starts a timer for the specified metric
  /// Can track multiple concurrent timers for the same metric
  /// - Parameter metric: The metric to start timing
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
  
  /// Ends timing for the specified metric and calculates average duration
  /// Automatically adds the calculated time to the metrics collection
  /// - Parameter metric: The metric to stop timing
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
  
  /// Updates a metric with a new value
  /// Internal method used by optimizers and models
  /// - Parameters:
  ///   - metric: The metric to update
  ///   - value: The new metric value
  internal func update(metric: Metric, value: Tensor.Scalar) {
    addMetric(value: value, key: metric)
  }
  
  /// Reports metrics to the registered callback if frequency condition is met
  /// Called internally to trigger metric reporting at specified intervals
  internal func report() {
    currentStep += 1
    if currentStep % frequency == 0 {
      receive?(metrics)
      currentStep = 0
    }
  }
}
