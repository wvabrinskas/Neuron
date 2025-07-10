//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation
import NumSwift

/// Protocol defining the interface for neural network optimizers
/// Optimizers are responsible for updating model parameters based on computed gradients
public protocol Optimizer: AnyObject {
  /// Gradient tuple containing weights and biases
  typealias Gradient = (weights: Tensor, biases: Tensor)
  /// Output tuple containing predictions, gradients, loss, and accuracy
  typealias Output = (outputs: [Tensor], gradients: Tensor.Gradient, loss: Tensor.Scalar, accuracy: Tensor.Scalar)
  
  /// The trainable model being optimized
  var trainable: Trainable { get set }
  /// Current learning rate (may be modified by decay functions)
  var learningRate: Tensor.Scalar { get }
  /// Whether the optimizer is in training mode
  var isTraining: Bool { get set }
  /// Computation device (CPU/GPU)
  var device: Device { get set }
  /// Whether to apply L2 normalization to gradients
  var l2Normalize: Bool { get }
  /// Optional metrics reporter for tracking training progress
  var metricsReporter: MetricsReporter? { get set }
  /// Optional gradient clipping threshold
  var clip: Tensor.Scalar? { get set }
  /// Accumulator for batching gradients
  var gradientAccumulator: GradientAccumulator { get }
  /// Optional learning rate decay function
  var decayFunction: DecayFunction? { get set }

  /// Allows optimizer to be called as a function for predictions
  /// - Parameter data: Input tensors to process
  /// - Returns: Prediction outputs
  func callAsFunction(_ data: [Tensor]) -> [Tensor]
  
  /// Applies gradients to the optimizer's gradient accumulator
  /// - Parameter gradients: Gradients to apply
  func apply(_ gradients: Tensor.Gradient)
  
  /// Clears accumulated gradients
  func zeroGradients()
  
  /// Performs one optimization step (parameter update)
  func step()
  
  /// Resets the optimizer state
  func reset()
  
  /// Fits the model on a batch of data
  /// - Parameters:
  ///   - data: Input tensors
  ///   - labels: Target labels
  ///   - lossFunction: Loss function to use
  ///   - validation: Whether this is validation data
  ///   - requiresGradients: Whether to compute gradients
  /// - Returns: Output containing predictions, gradients, loss, and accuracy
  func fit(_ data: [Tensor],
           labels: [Tensor],
           lossFunction: LossFunction,
           validation: Bool,
           requiresGradients: Bool) -> Output
  
  /// Makes predictions on input data
  /// - Parameter data: Input tensors to process
  /// - Returns: Prediction outputs
  func predict(_ data: [Tensor]) -> [Tensor]
}

/// Base implementation of the Optimizer protocol
/// Provides common functionality for all optimizer implementations
/// TODO: allow for arbitrary weight shape in Optimizer, so we dont have to cram all weights into a 3D tensor
open class BaseOptimizer: Optimizer {
  /// Optional learning rate decay function
  public var decayFunction: DecayFunction?
  /// The trainable model being optimized
  public var trainable: Trainable
  /// Current learning rate (modified by decay function if present)
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
  /// Whether the optimizer is in training mode
  public var isTraining: Bool = true {
    didSet {
      trainable.isTraining = isTraining
    }
  }
  /// Computation device (CPU/GPU)
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
  /// Whether to apply L2 normalization to gradients
  public var l2Normalize: Bool
  /// Optional metrics reporter for tracking training progress
  public var metricsReporter: MetricsReporter?
  /// Accumulator for batching gradients
  public var gradientAccumulator: GradientAccumulator = .init()
  /// Optional gradient clipping threshold
  public var clip: Tensor.Scalar?
  /// Local storage for learning rate (not affected by decay function)
  private var localLearningRate: Tensor.Scalar
  
  /// Initializes a base optimizer with the given parameters
  /// - Parameters:
  ///   - trainable: The model to optimize
  ///   - learningRate: Initial learning rate
  ///   - l2Normalize: Whether to apply L2 normalization to gradients
  ///   - metricsReporter: Optional metrics reporter
  ///   - clip: Optional gradient clipping threshold
  public init(trainable: Trainable,
              learningRate: Tensor.Scalar,
              l2Normalize: Bool,
              metricsReporter: MetricsReporter? = nil,
              clip: Tensor.Scalar? = nil) {
    self.trainable = trainable
    self.l2Normalize = l2Normalize
    self.metricsReporter = metricsReporter
    self.clip = clip
    self.localLearningRate = learningRate
    self.learningRate = learningRate
  }
  
  /// Performs one optimization step - updates parameters and advances decay function
  /// This method should be overridden by subclasses to implement specific optimization logic
  public func step() {
    // override
    decayFunction?.step()
    metricsReporter?.endTimer(metric: .optimizerRunTime)
  }
  
  /// Resets the optimizer state including decay function
  /// This method should be overridden by subclasses to reset optimizer-specific state
  public func reset() {
    // override
    decayFunction?.reset()
  }
  
  /// Clips gradients for a layer to prevent exploding gradients
  /// - Parameter layer: The layer whose gradients should be clipped
  func clip(layer: Layer) {
    if let clip = clip {
      if let con = layer as? ConvolutionalLayer {
        con.filters.forEach { $0.clip(clip) }
      } else {
        layer.weights.clip(clip)
      }
    }
  }
  
  /// Clears accumulated gradients and starts timing for optimizer runtime
  open func zeroGradients() {
    metricsReporter?.startTimer(metric: .optimizerRunTime)
    gradientAccumulator.clear()
  }
  
  /// Adds gradients to the optimizer's gradient accumulator without averaging
  /// Use a `GradientAccumulator` to get averaged gradients before applying them
  /// - Parameter newGradients: Gradients to add to the optimizer
  open func apply(_ newGradients: Tensor.Gradient) {
    gradientAccumulator.insert(newGradients)
  }
  
  /// Allows the optimizer to be called as a function for making predictions
  /// - Parameter data: Input tensors to process
  /// - Returns: Prediction outputs from the model
  open func callAsFunction(_ data: [Tensor]) -> [Tensor] {
    predict(data)
  }
  
  /// Makes predictions on input data using the trainable model
  /// Processes data concurrently across multiple workers for better performance
  /// - Parameter data: Input tensors to process
  /// - Returns: Prediction outputs from the model
  open func predict(_ data: [Tensor]) -> [Tensor] {
    var results: [Tensor] = [Tensor].init(repeating: Tensor(), count: data.count)

    data.concurrentForEach(workers: Constants.maxWorkers,
                           priority: device.qosPriority) { tensor, index in
      let output = self.trainable.predict(tensor, context: .init(threadId: index))
      results[index] = output
    }

    return results
  }
  
  /// Fits the model on a batch of training data
  /// Performs forward pass, loss calculation, and optional gradient computation
  /// - Parameters:
  ///   - data: Input tensors for training
  ///   - labels: Target labels for supervised learning
  ///   - lossFunction: Loss function to use for training
  ///   - validation: Whether this is validation data (affects accuracy calculation)
  ///   - requiresGradients: Whether to compute gradients for backpropagation
  /// - Returns: Output containing predictions, gradients, loss, and accuracy
  open func fit(_ data: [Tensor],
                  labels: [Tensor],
                  lossFunction: LossFunction,
                  validation: Bool = false,
                  requiresGradients: Bool = true) -> Output {
    
    metricsReporter?.startTimer(metric: .batchTime)
    let accumulator = GradientAccumulator()
    
    var outputs: [Tensor] = [Tensor].init(repeating: Tensor(), count: data.count)
    
    var losses: Tensor.Scalar = 0
    var accuracy: Tensor.Scalar = 0
    
    // TODO: Batch consolidation: https://github.com/wvabrinskas/Neuron/issues/36
    
    let workersCount = Constants.maxWorkers
    let concurrencySplit = Tensor.Scalar(data.count) / Tensor.Scalar(workersCount)
    
    metricsReporter?.update(metric: .batchConcurrency, value: concurrencySplit)
  
    data.concurrentForEach(workers: workersCount, priority: device.qosPriority) { b, index in
      let label: [Tensor.Scalar] = labels[index].value.flatten()
      let input = data[index]
      
      let out = self.trainable.predict(input, context: .init(threadId: index))
      
      outputs[index] = out
            
      let loss = lossFunction.calculate(out, correct: labels[index]).sum(axis: -1).asScalar()
      losses += loss / Tensor.Scalar(data.count)
      
      if let reporter = self.metricsReporter {
        if validation {
          accuracy += reporter.calculateValAccuracy(out, label: labels[index], binary: label.count == 1, running: false) / Tensor.Scalar(data.count)
        } else {
          accuracy += reporter.calculateAccuracy(out, label: labels[index], binary: label.count == 1, running: false) / Tensor.Scalar(data.count)
        }
      }
      
      if requiresGradients {
        let lossGradient = lossFunction.derivative(out, correct: labels[index])
        let gradient = out.gradients(delta: lossGradient)
        accumulator.insert(gradient)
      }
    }
    
    metricsReporter?.endTimer(metric: .batchTime)
    
    if requiresGradients {
      let accumulated = accumulator.accumulate(clearAtEnd: true)
      let weightGradientsAcc: [Tensor] = accumulated.weights
      let inputGradientsAcc: [Tensor] = accumulated.input
      let biasGradientAcc: [Tensor] = accumulated.biases
      
      let gradient: Tensor.Gradient = .init(input: inputGradientsAcc,
                                            weights: weightGradientsAcc,
                                            biases: biasGradientAcc)
      return (outputs, gradient, losses, accuracy)
      
    } else {
      let gradient: Tensor.Gradient = .init(input: [],
                                            weights: [],
                                            biases: [])
      
      return (outputs, gradient, losses, accuracy)
    }
  }
}

