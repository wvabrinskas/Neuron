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
  var l2Normalize: Bool { get }
  var metricsReporter: MetricsReporter? { get set }
  var clip: Tensor.Scalar? { get set }
  var gradientAccumulator: GradientAccumulator { get }
  var decayFunction: DecayFunction? { get set }

  func callAsFunction(_ data: [Tensor]) -> [Tensor]
  func apply(_ gradients: Tensor.Gradient)
  func zeroGradients()
  func step()
  func reset()
  func fit(_ data: [Tensor],
           labels: [Tensor],
           lossFunction: LossFunction,
           validation: Bool,
           requiresGradients: Bool) -> Output
  func predict(_ data: [Tensor]) -> [Tensor]
}

// TODO: allow for arbitrary weight shape in Optimizer, so we dont have to cram all weights into a 3D tensor
open class BaseOptimizer: Optimizer {
  public var decayFunction: DecayFunction?
  public var trainable: Trainable
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
  public var l2Normalize: Bool
  public var metricsReporter: MetricsReporter?
  public var gradientAccumulator: GradientAccumulator = .init()
  public var clip: Tensor.Scalar?
  private var localLearningRate: Tensor.Scalar
  
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
  
  public func step() {
    // override
    decayFunction?.step()
    metricsReporter?.endTimer(metric: .optimizerRunTime)
  }
  
  public func reset() {
    // override
    decayFunction?.reset()
  }
  
  func clip(layer: Layer) {
    if let clip = clip {
      if let con = layer as? ConvolutionalLayer {
        con.filters.forEach { $0.clip(clip) }
      } else {
        layer.weights.clip(clip)
      }
    }
  }
  
  open func zeroGradients() {
    metricsReporter?.startTimer(metric: .optimizerRunTime)
    gradientAccumulator.clear()
  }
  
  /// Adds gradients to the Optimizer but does not provide an average. Please use a `GradientAccumulator` to get average gradients then apply them.
  /// - Parameter newGradients: Gradients to add to the `Optimizer`
  open func apply(_ newGradients: Tensor.Gradient) {
    gradientAccumulator.insert(newGradients)
  }
  
  open func callAsFunction(_ data: [Tensor]) -> [Tensor] {
    predict(data)
  }
  
  open func predict(_ data: [Tensor]) -> [Tensor] {
    var results: [Tensor] = [Tensor].init(repeating: Tensor(), count: data.count)

    data.concurrentForEach(workers: Constants.maxWorkers,
                           priority: device.qosPriority) { tensor, index in
      let output = self.trainable.predict(tensor, context: .init(threadId: index))
      results[index] = output
    }

    return results
  }
  
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

