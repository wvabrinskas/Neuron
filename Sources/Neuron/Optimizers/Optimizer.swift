//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation
import NumSwift

public protocol Optimizer: AnyObject {
  typealias Gradient = (weights: Tensor, biases: Tensor)
  typealias Output = (outputs: [Tensor], gradients: Tensor.Gradient, loss: Tensor.Scalar, accuracy: Tensor.Scalar)
  
  var trainable: Trainable { get set }
  var learningRate: Tensor.Scalar { get }
  var isTraining: Bool { get set }
  var device: Device { get set }
  var metricsReporter: MetricsReporter? { get set }
  var weightClip: Tensor.Scalar? { get set }
  var gradientClip: Tensor.Scalar? { get set }
  var gradientAccumulator: GradientAccumulator { get }
  var decayFunction: DecayFunction? { get set }
  var batchSize: Int { get }
  var augmenter: Augmenter? { get set }

  /// Runs inference via call syntax.
  ///
  /// - Parameter data: Input tensor batch.
  /// - Returns: Prediction tensors.
  func callAsFunction(_ data: [Tensor]) -> [Tensor]
  /// Adds gradients to the optimizer's accumulator/state.
  ///
  /// - Parameter gradients: Newly computed gradients to accumulate.
  func apply(_ gradients: Tensor.Gradient)
  /// Clears any currently accumulated gradients.
  func zeroGradients()
  /// Applies one optimization step using accumulated gradients.
  func step()
  /// Resets optimizer-specific running state.
  func reset()
  /// Performs one fit iteration on a batch.
  ///
  /// - Parameters:
  ///   - data: Input tensors.
  ///   - labels: Ground-truth labels for `data`.
  ///   - wrt: Optional explicit gradient target tensors.
  ///   - lossFunction: Loss function used for optimization.
  ///   - validation: Whether this run is validation-only.
  ///   - requiresGradients: Whether gradients should be computed.
  /// - Returns: Outputs, gradients, loss, and accuracy for the batch.
  func fit(_ data: [Tensor],
           labels: [Tensor],
           wrt: TensorBatch?,
           lossFunction: LossFunction,
           validation: Bool,
           requiresGradients: Bool) -> Output
  /// Runs prediction for a batch without training updates.
  ///
  /// - Parameter data: Input tensors.
  /// - Returns: Prediction tensors.
  func predict(_ data: [Tensor]) -> [Tensor]
}

// TODO: allow for arbitrary weight shape in Optimizer, so we dont have to cram all weights into a 3D tensor
open class BaseOptimizer: Optimizer {
  public var augmenter: Augmenter?
  public var decayFunction: DecayFunction?
  public var trainable: Trainable
  public var batchSize: Int
  public var passthroughGradientCalculation: Bool = false
  public var learningRate: Tensor.Scalar {
    get {
      if let decayFunction {
        return decayFunction.decayedLearningRate
      } else {
        return localLearningRate
      }
    }
    set {
      localLearningRate = newValue
    }
  }
  public var isTraining: Bool = true {
    didSet {
      trainable.isTraining = isTraining
    }
  }
  public var device: Device = CPU() {
    didSet {
      switch device.type {
      case .cpu:
        trainable.device = CPU()
      case .gpu:
        trainable.device = GPU()
      }
    }
  }

  public var metricsReporter: MetricsReporter?
  public var gradientAccumulator: GradientAccumulator = .init()
  public var weightClip: Tensor.Scalar?
  public var gradientClip: Tensor.Scalar?
  private var localLearningRate: Tensor.Scalar
  private let workersCount = Constants.maxWorkers
  private var augmentation: Augmenting? = nil

  /// Creates an optimizer base configured to train a `Trainable` network.
  ///
  /// - Parameters:
  ///   - trainable: Network whose layer parameters will be updated.
  ///   - learningRate: Base learning rate (unless a decay function overrides it).
  ///   - batchSize: Number of samples processed per optimization step.
  ///   - metricsReporter: Optional reporter that collects loss/timing metrics.
  ///   - weightClip: Optional per-layer weight clipping threshold.
  ///   - gradientClip: Optional gradient clipping threshold.
  ///   - augmenter: Optional data augmentation policy applied during training.
  public init(trainable: Trainable,
              learningRate: Tensor.Scalar,
              batchSize: Int,
              metricsReporter: MetricsReporter? = nil,
              weightClip: Tensor.Scalar? = nil,
              gradientClip: Tensor.Scalar? = nil,
              augmenter: Augmenter? = nil) {
    self.trainable = trainable
    self.metricsReporter = metricsReporter
    self.weightClip = weightClip
    self.gradientClip = gradientClip
    self.localLearningRate = learningRate
    self.batchSize = batchSize
    self.learningRate = learningRate
    self.augmenter = augmenter
    trainable.batchSize = batchSize
    
    self.augmentation = augmenter?.augmenting
  }
  
  /// Finalizes an optimization step after gradients have been applied.
  ///
  /// Subclasses should override to implement algorithm-specific updates, then
  /// call `super.step()` to advance decay/timer bookkeeping.
  public func step() {
    // override
    decayFunction?.step()
    metricsReporter?.endTimer(metric: .optimizerRunTime)
  }
  
  /// Resets optimizer-managed state (for example momentum/Adam moments).
  ///
  /// Subclasses should clear algorithm-specific buffers, then call `super.reset()`.
  public func reset() {
    // override
    decayFunction?.reset()
  }
  
  func gradientClip(_ gradients: Optimizer.Gradient) {
    if let clip = gradientClip {
      gradients.biases.clip(clip)
      gradients.weights.clip(clip)
    }
  }
  
  func weightClip(layer: Layer) {
    if let clip = weightClip {
      if let con = layer as? ConvolutionalLayer {
        con.filters.forEach { $0.clip(clip) }
      } else {
        layer.weights.clip(clip)
      }
    }
  }
  
  /// Clears all accumulated gradients before processing a new batch.
  open func zeroGradients() {
    metricsReporter?.startTimer(metric: .optimizerRunTime)
    gradientAccumulator.clear()
  }
  
  /// Adds gradients to the Optimizer but does not provide an average. Please use a `GradientAccumulator` to get average gradients then apply them.
  /// - Parameter newGradients: Gradients to add to the `Optimizer`
  open func apply(_ newGradients: Tensor.Gradient) {
    gradientAccumulator.insert(newGradients)
  }
  
  /// Runs inference on a tensor batch via call syntax.
  ///
  /// - Parameter data: Batch of input tensors.
  /// - Returns: Predictions in input order.
  open func callAsFunction(_ data: [Tensor]) -> [Tensor] {
    predict(data)
  }
  
  /// Performs batched forward prediction without parameter updates.
  ///
  /// - Parameter data: Batch of input tensors.
  /// - Returns: Forward outputs from the underlying trainable.
  open func predict(_ data: [Tensor]) -> [Tensor] {
    isTraining = false
    
    var results: [Tensor] = [Tensor].init(repeating: Tensor(), count: data.count)

    data.concurrentForEach(workers: Constants.maxWorkers,
                           priority: device.qosPriority) { tensor, index in
      let output = self.trainable.predict(tensor, context: .init(indexInBatch: index))
      results[index] = output
    }

    return results
  }
  
  /// Runs one training/validation iteration over the provided batch.
  ///
  /// - Parameters:
  ///   - data: Input tensors.
  ///   - labels: Ground-truth tensors aligned with `data`.
  ///   - wrt: Optional tensors specifying explicit backpropagation targets.
  ///   - lossFunction: Loss used for value and derivative computation.
  ///   - validation: Flag indicating validation mode (no parameter updates).
  ///   - requiresGradients: Whether to compute and return gradients.
  /// - Returns: Batch outputs, aggregated gradients, loss, and accuracy.
  open func fit(_ data: TensorBatch,
                labels: TensorBatch,
                wrt: TensorBatch? = nil,
                lossFunction: LossFunction,
                validation: Bool = false,
                requiresGradients: Bool = true) -> Output {
          
    if let wrt {
      guard wrt.count == data.count else {
        fatalError("The number of wrt inputs (\(wrt.count)) does not match the number of training examples (\(data.count)).")
      }
    }
    
    isTraining = !validation
        
    metricsReporter?.startTimer(metric: .batchTime)
   // let accumulator = GradientAccumulator()
  
    var losses: Tensor.Scalar = 0
    var accuracy: Tensor.Scalar = 0
        
    let concurrencySplit = Tensor.Scalar(data.count) / Tensor.Scalar(workersCount)
    
    var outputs: [[Tensor]] = [[Tensor]].init(repeating: [], count: workersCount)

    metricsReporter?.update(metric: .batchConcurrency, value: concurrencySplit)
    
    var dataToUse = data
    var augmentedOut: AugementedDatasetModel?
    
    if let augmentation, validation == false {
      let augOut = augmentation.augment(dataToUse, labels: labels)
      dataToUse = augOut.mixed
      augmentedOut = augOut
    }
    
    var accumulators = (0..<workersCount).map { _ in GradientAccumulator() }
    
    dataToUse.concurrentBatchedForEach(workers: workersCount, priority: device.qosPriority) { elements,
                                                                                         workerIndex,
                                                                                         indexRange,
                                                                                         processingCount,
                                                                                         workerId in
      
      let accumulator = accumulators[workerIndex]
      accumulator.average = false

      let outs = self.trainable.predict(batch: elements, context: .init(batchRange: indexRange,
                                                                        batchProcessingCount: processingCount,
                                                                        totalInBatch: data.count,
                                                                        threadId: workerId))
      
      outputs[workerIndex] = outs
      
      let wrtBatch: TensorBatch? = if let wrt {
        Array(wrt[indexRange])
      } else {
        nil
      }
      
      var batchLabels: [Tensor] = Array(labels[indexRange])
      
      if let mixedLabels = augmentedOut?.mixedLabels {
        batchLabels = Array(mixedLabels[indexRange])
      }

      for (index, out) in outs.enumerated() {
        let label = batchLabels[index]
        let input = wrtBatch?[index] ?? elements[index]
        
        let loss = lossFunction.calculate(out, correct: label).sum(axis: -1)
        
        losses += loss.asScalar() / Tensor.Scalar(data.count)
        
        if let reporter = self.metricsReporter {
          if validation {
            accuracy += reporter.calculateValAccuracy(out, label: label, binary: label.isScalar(), running: false) / Tensor.Scalar(data.count)
          } else {
            let localAccuracy = reporter.calculateAccuracy(out, label: label, binary: label.isScalar(), running: false)
            accuracy += localAccuracy / Tensor.Scalar(data.count)
          }
        }
        
        if requiresGradients {
          let lossGradient = lossFunction.derivative(out, correct: label)
          let gradient = out.gradients(delta: lossGradient, wrt: input)
          accumulator.insert(gradient)
        }
      }
    }
    
    metricsReporter?.endTimer(metric: .batchTime)

    let flatOutput = outputs.flatMap { $0 }

    if requiresGradients {
      
      // account for too many accumulators. we allocate for each thread but that might not be needed
      accumulators = accumulators.compactMap { $0.isEmpty ? nil : $0 }
      
      var accumulatedGradients = accumulators.map { $0.accumulate() }
      let first = accumulatedGradients.removeFirst()
      
      let accumulated = accumulatedGradients.reduce(first, +) / batchSize.asTensorScalar
      
      let weightGradientsAcc: [Tensor] = accumulated.weights
      let inputGradientsAcc: [Tensor] = accumulated.input
      let biasGradientAcc: [Tensor] = accumulated.biases
      
      let gradient: Tensor.Gradient = .init(input: inputGradientsAcc,
                                            weights: weightGradientsAcc,
                                            biases: biasGradientAcc)
      return (flatOutput, gradient, losses, accuracy)
      
    } else {
      let gradient: Tensor.Gradient = .init(input: [],
                                            weights: [],
                                            biases: [])
      
      return (flatOutput, gradient, losses, accuracy)
    }
  }
  
}

