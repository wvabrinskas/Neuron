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
  /// Training loss computed over the current batch.
  case loss
  /// Training accuracy computed over the current batch.
  case accuracy
  /// Validation loss computed on the held-out validation set.
  case valLoss
  /// Generator loss for GAN-style training.
  case generatorLoss
  /// Critic (discriminator) loss for WGAN/GAN-style training.
  case criticLoss
  /// Gradient penalty term used in WGAN-GP training.
  case gradientPenalty
  /// Discriminator loss on real samples.
  case realImageLoss
  /// Discriminator loss on fake (generated) samples.
  case fakeImageLoss
  /// Validation accuracy computed on the held-out validation set.
  case valAccuracy
  /// Wall-clock time taken to process one training batch (forward + backward + accumulation).
  case batchTime
  /// Wall-clock time taken by the optimizer's `step()` call (gradient application).
  case optimizerRunTime
  /// Average number of samples processed concurrently per worker.
  case batchConcurrency
  /// Global L2 norm of all gradient tensors before clipping.
  case globalGradientNorm
  /// Scaling factor applied to gradients during global norm clipping.
  case globalGradientScalingFactor
  /// The current effective learning rate output by the active schedule.
  case currentLearningRate
}

/// A protocol that defines the interface for objects that collect and store training metrics.
public protocol MetricLogger: AnyObject {
  /// The set of metrics actively gathered by this logger.
  var metricsToGather: Set<Metric> { get set }
  /// The current scalar values for each metric, keyed by `Metric`.
  var metrics: [Metric: Tensor.Scalar] { get set }
  /// A lock used to serialize concurrent metric writes.
  var lock: NSLock { get }
  /// Records a metric value when the metric is enabled for gathering.
  ///
  /// - Parameters:
  ///   - value: Metric value to store.
  ///   - key: Metric key identifying the value.
  func addMetric(value: Tensor.Scalar, key: Metric)
}

/// Default implementations for the `MetricLogger` protocol.
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

  func calculateAccuracy(_ guess: Tensor, label: Tensor, binary: Bool, sparse: Bool, ignoring: Int?, running: Bool) -> Tensor.Scalar
  func calculateValAccuracy(_ guess: Tensor, label: Tensor, binary: Bool, sparse: Bool, ignoring: Int?, running: Bool) -> Tensor.Scalar
  func startTimer(metric: Metric)
  func endTimer(metric: Metric)
}

internal extension MetricCalculator {
  typealias Max = (UInt, Tensor.Scalar)

  /// Resolves the target `(index, value)` pair for depth slice `d` of a label tensor.
  ///
  /// One-hot labels carry the class in the *position* of their maximum, sparse labels carry it
  /// as the *value* of their single entry, so `indexOfMax` is only correct for the former.
  private func labelMax(_ label: Tensor, depth d: Int, sparse: Bool) -> Max {
    guard sparse else { return label.depthSlice(d).indexOfMax }

    let index = label.storage.count > d ? Int(label.storage[d]) : 0
    return (UInt(max(0, index)), Tensor.Scalar(index))
  }

  private func matches(label: Max, guess: Max, binary: Bool) -> Int {
    if binary {
      if label.1 - guess.1 < 0.5 { return 1 }
    } else {
      if label.0 == guess.0 { return 1 }
    }
    return 0
  }

  /// Whether depth slice `d` should be scored at all.
  ///
  /// Padded timesteps are trivially predictable, so counting them reports an accuracy that
  /// mostly measures how much padding the batch carried.
  private func isScored(_ label: Tensor, depth d: Int, sparse: Bool, ignoring: Int?) -> Bool {
    guard sparse, let ignoring, label.storage.count > d else { return true }
    return Int(label.storage[d]) != ignoring
  }

  func calculateValAccuracy(_ guess: Tensor, label: Tensor, binary: Bool, sparse: Bool = false, ignoring: Int? = nil, running: Bool = false) -> Tensor.Scalar {
    var totalCorrect = 0
    var totalGuess = 0

    for d in 0..<guess.size.depth {
      guard isScored(label, depth: d, sparse: sparse, ignoring: ignoring) else { continue }

      let guessMax = guess.depthSlice(d).indexOfMax
      let correct = matches(label: labelMax(label, depth: d, sparse: sparse),
                            guess: guessMax,
                            binary: binary)
      totalCorrect += correct
      totalValCorrectGuesses += correct
      totalGuess += 1
      totalValGuesses += 1
    }
    
    let runningAccuracy = totalValGuesses > 0
      ? Tensor.Scalar(totalValCorrectGuesses) / Tensor.Scalar(totalValGuesses) * 100.0
      : 0
    let accuracy = totalGuess > 0
      ? Tensor.Scalar(totalCorrect) / Tensor.Scalar(totalGuess) * 100.0
      : 0
    return running ? runningAccuracy : accuracy
  }
  
  func calculateAccuracy(_ guess: Tensor, label: Tensor, binary: Bool, sparse: Bool = false, ignoring: Int? = nil, running: Bool = false) -> Tensor.Scalar {
    var totalCorrect = 0
    var totalGuess = 0

    for d in 0..<guess.size.depth {
      guard isScored(label, depth: d, sparse: sparse, ignoring: ignoring) else { continue }

      let guessMax = guess.depthSlice(d).indexOfMax
      let correct = matches(label: labelMax(label, depth: d, sparse: sparse),
                            guess: guessMax,
                            binary: binary)
      totalCorrect += correct
      totalCorrectGuesses += correct
      totalGuess += 1
      totalGuesses += 1
    }
    
    let runningAccuracy = totalGuesses > 0
      ? Tensor.Scalar(totalCorrectGuesses) / Tensor.Scalar(totalGuesses) * 100.0
      : 0
    let accuracy = totalGuess > 0
      ? Tensor.Scalar(totalCorrect) / Tensor.Scalar(totalGuess) * 100.0
      : 0
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
