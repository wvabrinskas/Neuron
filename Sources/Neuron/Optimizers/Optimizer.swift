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

  func callAsFunction(_ data: [Tensor]) -> [Tensor]
  func apply(_ gradients: Tensor.Gradient)
  func zeroGradients()
  func step()
  func reset()
  func fit(_ data: [Tensor],
           labels: [Tensor],
           wrt: TensorBatch?,
           lossFunction: LossFunction,
           validation: Bool,
           requiresGradients: Bool) -> Output
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
  
  public func step() {
    // override
    decayFunction?.step()
    metricsReporter?.endTimer(metric: .optimizerRunTime)
  }
  
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
    isTraining = false
    
    var results: [Tensor] = [Tensor].init(repeating: Tensor(), count: data.count)

    data.concurrentForEach(workers: Constants.maxWorkers,
                           priority: device.qosPriority) { tensor, index in
      let output = self.trainable.predict(tensor, context: .init(indexInBatch: index))
      results[index] = output
    }

    return results
  }
  
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
    let accumulator = GradientAccumulator()
  
    var losses: Tensor.Scalar = 0
    var accuracy: Tensor.Scalar = 0
        
    let concurrencySplit = Tensor.Scalar(data.count) / Tensor.Scalar(workersCount)
    
    var outputs: [[Tensor]] = [[Tensor]].init(repeating: [], count: workersCount)

    metricsReporter?.update(metric: .batchConcurrency, value: concurrencySplit)
    
    var dataToUse = data
    var augmentedOut: AugementedDatasetModel?
    
    if let augmentation {
      let augOut = augmentation.augment(dataToUse, labels: labels)
      dataToUse = augOut.mixed
      augmentedOut = augOut
    }
    
    dataToUse.concurrentBatchedForEach(workers: workersCount, priority: device.qosPriority) { elements,
                                                                                         workerIndex,
                                                                                         indexRange,
                                                                                         processingCount,
                                                                                         workerId in
      
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
    
    let flatOutput = outputs.flatMap { $0 }
    
    metricsReporter?.endTimer(metric: .batchTime)
    
    if requiresGradients {
      let accumulated = accumulator.accumulate(clearAtEnd: true)
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

