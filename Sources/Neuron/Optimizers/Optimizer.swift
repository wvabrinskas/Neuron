//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation
import NumSwift

public protocol Optimizer: AnyObject {
  associatedtype N: TensorNumeric
  typealias Gradient = (weights: Tensor<N>, biases: Tensor<N>)
  typealias Output = (outputs: [Tensor<N>], gradients: Tensor<N>.Gradient, loss: Tensor<N>.Scalar, accuracy: Tensor<N>.Scalar)
  
  var trainable: BaseTrainable<N> { get set }
  var learningRate: Tensor<N>.Scalar { get }
  var isTraining: Bool { get set }
  var device: BaseDevice<N> { get set }
  var l2Normalize: Bool { get }
  var workers: Int { get set }
  var metricsReporter: MetricsReporter<N>? { get set }
  var clip: Tensor<N>.Scalar? { get set }
  var gradientAccumulator: GradientAccumulator<N> { get }
  var decayFunction: BaseDecayFunction<N>? { get set }

  func callAsFunction(_ data: [Tensor<N>]) -> [Tensor<N>]
  func apply(_ gradients: Tensor<N>.Gradient)
  func zeroGradients()
  func step()
  func reset()
  func fit(_ data: [Tensor<N>],
           labels: [Tensor<N>],
           lossFunction: LossFunction,
           validation: Bool,
           requiresGradients: Bool) -> Output
  func predict(_ data: [Tensor<N>]) -> [Tensor<N>]
}

// TODO: allow for arbitrary weight shape in Optimizer, so we dont have to cram all weights into a 3D tensor
open class BaseOptimizer<N: TensorNumeric>: Optimizer {
  public var decayFunction: BaseDecayFunction<N>?
  public var trainable: BaseTrainable<N>
  public var learningRate: Tensor<N>.Scalar {
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
  public var device: BaseDevice<N> = CPU<N>() {
    didSet {
      switch device.type {
      case .cpu:
        trainable.device = CPU<N>()
      case .gpu:
        trainable.device = GPU<N>()
      }
    }
  }
  public var l2Normalize: Bool
  public var workers: Int
  public var metricsReporter: MetricsReporter<N>?
  public var gradientAccumulator: GradientAccumulator<N> = .init()
  public var clip: Tensor<N>.Scalar?
  private var localLearningRate: Tensor<N>.Scalar
  
  public init(trainable: BaseTrainable<N>,
              learningRate: Tensor<N>.Scalar,
              l2Normalize: Bool,
              workers: Int = 8,
              metricsReporter: MetricsReporter<N>? = nil,
              clip: Tensor<N>.Scalar? = nil) {
    self.trainable = trainable
    self.l2Normalize = l2Normalize
    self.workers = workers
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
  
  func clip(layer: BaseLayer<N>) {
    if let clip = clip {
      if let con = layer as? BaseConvolutionalLayer<N> {
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
  open func apply(_ newGradients: Tensor<N>.Gradient) {
    gradientAccumulator.insert(newGradients)
  }
  
  open func callAsFunction(_ data: [Tensor<N>]) -> [Tensor<N>] {
    predict(data)
  }
  
  open func predict(_ data: [Tensor<N>]) -> [Tensor<N>] {
    var results: [Tensor<N>] = [Tensor<N>].init(repeating: Tensor<N>(), count: data.count)

    data.concurrentForEach(workers: min(Constants.maxWorkers, Int(ceil(Double(data.count) / Double(4)))),
                           priority: device.qosPriority) { tensor, index in
      let output = self.trainable.predict(tensor)
      results[index] = output
    }

    return results
  }
  
  open func fit(_ data: [Tensor<N>],
                  labels: [Tensor<N>],
                  lossFunction: LossFunction,
                  validation: Bool = false,
                  requiresGradients: Bool = true) -> Output {
    
    metricsReporter?.startTimer(metric: .batchTime)
    let accumulator = GradientAccumulator<N>()
    
    var outputs: [Tensor<N>] = [Tensor<N>].init(repeating: Tensor<N>(), count: data.count)
    
    var losses: Tensor<N>.Scalar = 0
    var accuracy: Tensor<N>.Scalar = 0
    
    // TODO: Batch consolidation: https://github.com/wvabrinskas/Neuron/issues/36
    let workersCount = min(Constants.maxWorkers, workers)
    let concurrencySplit = Tensor<N>.Scalar(data.count) / Tensor<N>.Scalar(workersCount)
    metricsReporter?.update(metric: .batchConcurrency, value: concurrencySplit)
  
    data.concurrentForEach(workers: workersCount, priority: device.qosPriority) { b, index in
      let label: [Tensor<N>.Scalar] = labels[index].value.flatten()
      let input = data[index]
      
      self.trainable.threadId = index
      let out = self.trainable.predict(input)
      
      outputs[index] = out
            
      let loss = lossFunction.calculate(out, correct: labels[index]).sum(axis: -1).asScalar()
      losses += loss / Tensor<N>.Scalar(data.count)
      
      if let reporter = self.metricsReporter {
        if validation {
          accuracy += reporter.calculateValAccuracy(out, label: labels[index], binary: label.count == 1, running: false) / Tensor<N>.Scalar(data.count)
        } else {
          accuracy += reporter.calculateAccuracy(out, label: labels[index], binary: label.count == 1, running: false) / Tensor<N>.Scalar(data.count)
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
      let weightGradientsAcc: [Tensor<N>] = accumulated.weights
      let inputGradientsAcc: [Tensor<N>] = accumulated.input
      let biasGradientAcc: [Tensor<N>] = accumulated.biases
      
      let gradient: Tensor<N>.Gradient = .init(input: inputGradientsAcc,
                                            weights: weightGradientsAcc,
                                            biases: biasGradientAcc)
      return (outputs, gradient, losses, accuracy)
      
    } else {
      let gradient: Tensor<N>.Gradient = .init(input: [],
                                            weights: [],
                                            biases: [])
      
      return (outputs, gradient, losses, accuracy)
    }
  }
}

