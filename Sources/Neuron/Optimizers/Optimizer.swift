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
  typealias Output = (outputs: [Tensor], gradients: Tensor.Gradient, loss: Float, accuracy: Float)
  
  var trainable: Trainable { get set }
  var learningRate: Float { get }
  var isTraining: Bool { get set }
  var device: Device { get set }
  var l2Normalize: Bool { get }
  var workers: Int { get set }
  var metricsReporter: MetricsReporter? { get set }
  var clip: Float? { get set }
  var gradientAccumulator: GradientAccumulator { get }

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
public extension Optimizer {
  func clip(layer: Layer) {
    if let clip = clip {
      if let con = layer as? ConvolutionalLayer {
        con.filters.forEach { $0.clip(clip) }
      } else {
        layer.weights.clip(clip)
      }
    }
  }
  
  func zeroGradients() {
    gradientAccumulator.clear()
  }
  
  /// Adds gradients to the Optimizer but does not provide an average. Please use a `GradientAccumulator` to get average gradients then apply them.
  /// - Parameter newGradients: Gradients to add to the `Optimizer`
  func apply(_ newGradients: Tensor.Gradient) {
    gradientAccumulator.insert(newGradients)
  }
  
  func callAsFunction(_ data: [Tensor]) -> [Tensor] {
    predict(data)
  }
  
  func predict(_ data: [Tensor]) -> [Tensor] {
    var results: [Tensor] = [Tensor].init(repeating: Tensor(), count: data.count)

    data.concurrentForEach(workers: Int(ceil(Double(data.count) / Double(4))),
                           priority: device.qosPriority) { tensor, index in
      let output = self.trainable.predict(tensor)
      results[index] = output
    }

    return results
  }
  
  func fit(_ data: [Tensor],
           labels: [Tensor],
           lossFunction: LossFunction,
           validation: Bool = false,
           requiresGradients: Bool = true) -> Output {
    
    let accumulator = GradientAccumulator()
    
    var outputs: [Tensor] = [Tensor].init(repeating: Tensor(), count: data.count)
    
    var losses: Tensor.Scalar = 0
    var accuracy: Tensor.Scalar = 0
    
    // TODO: Batch consolidation: https://github.com/wvabrinskas/Neuron/issues/36
  
    data.concurrentForEach(workers: workers, priority: device.qosPriority) { b, index in
      let label: [Tensor.Scalar] = labels[index].value.flatten()
      let input = data[index]
      
      let out = self.trainable.predict(input)
      
      outputs[index] = out

      let outFlat: [Tensor.Scalar] = out.value.flatten()
            
      let loss = lossFunction.calculate(outFlat, correct: label)
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
