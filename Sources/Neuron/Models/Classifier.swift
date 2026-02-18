//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/3/22.
//

import Foundation
import NumSwift

public class Classifier {
  private var batchSize: Int
  private let epochs: Int
  private let log: Bool
  private let lossFunction: LossFunction
  private let accuracyThreshold: AccuracyThreshold
  private let killOnAccuracy: Bool
  private let accuracyMonitor: AccuracyMonitor
  
  public var onAccuracyReached: (() -> ())? = nil
  public var onEpochCompleted: (() -> ())? = nil
  
  public private(set) var optimizer: Optimizer
  
  /// Creates a classifier training wrapper around an optimizer/trainable pair.
  ///
  /// - Parameters:
  ///   - optimizer: Optimizer responsible for updates and inference.
  ///   - epochs: Total epoch count for `fit`.
  ///   - batchSize: Mini-batch size used during training/validation.
  ///   - accuracyThreshold: Early-stop threshold configuration.
  ///   - killOnAccuracy: Stops training when threshold is reached.
  ///   - log: Enables per-batch progress printing.
  ///   - lossFunction: Loss used for optimization.
  public init(optimizer: Optimizer,
              epochs: Int = 100,
              batchSize: Int,
              accuracyThreshold: AccuracyThreshold = .init(value: 0.8, averageCount: 5),
              killOnAccuracy: Bool = true,
              log: Bool = false,
              lossFunction: LossFunction = .crossEntropySoftmax) {
    self.batchSize = batchSize
    self.accuracyThreshold = accuracyThreshold
    self.optimizer = optimizer
    self.epochs = epochs
    self.killOnAccuracy = killOnAccuracy
    self.log = log
    self.lossFunction = lossFunction
    self.accuracyMonitor = .init(threshold: accuracyThreshold)
  }
  
  /// Runs inference for a batch of tensors.
  ///
  /// - Parameter data: Input tensors.
  /// - Returns: Model predictions.
  public func feed(_ data: [Tensor]) -> [Tensor] {
    optimizer(data)
  }
  
  /// Trains the classifier for the configured number of epochs.
  ///
  /// - Parameters:
  ///   - data: Training dataset.
  ///   - validation: Validation dataset sampled each epoch.
  public func fit(_ data: [DatasetModel], _ validation: [DatasetModel]) {
    //shuffle data
    let shuffledData = data.shuffled()
    let trainingBatches = shuffledData
      .batched(into: batchSize)
      .map(splitDataset)
    
    let validationBatches = validation
      .batched(into: batchSize)
      .map(splitDataset)
    
    for i in 0..<epochs {
      let startTime = CFAbsoluteTimeGetCurrent()

      var b = 0
      
      if let randomValBatch = validationBatches.randomElement() {
        optimizer.isTraining = false

        let result = trainOn(randomValBatch.data,
                             labels: randomValBatch.labels,
                             validation: true,
                             requiresGradients: false)
        let loss = result.loss
        optimizer.metricsReporter?.update(metric: .valLoss, value: loss)
        optimizer.metricsReporter?.update(metric: .valAccuracy, value: result.accuracy)

        print("val_complete - ", "loss: ", loss, "accuracy: ", result.accuracy)
        
        accuracyMonitor.append(result.accuracy / 100.0)
        
        optimizer.isTraining = true
        
        if accuracyMonitor.isAboveThreshold() {
          self.onAccuracyReached?()
          if killOnAccuracy {
            return
          }
        }
        
      }
            
      for batch in trainingBatches {
        optimizer.zeroGradients()

        let result = trainOn(batch.data, labels: batch.labels) // multi threaded
        let weightGradients = result.gradients
        let loss = result.loss
        
        optimizer.metricsReporter?.update(metric: .loss, value: loss)
        optimizer.metricsReporter?.update(metric: .accuracy, value: result.accuracy)

        let batchesCompletePercent = (round((Double(b) / Double(trainingBatches.count)) * 10000) / 10000) * 100
        
        if log {
          print("complete :", "\(b) / \(trainingBatches.count) -> \(batchesCompletePercent)%")
        }
        
        optimizer.apply(weightGradients) // single threaded
        optimizer.step() // single threaded
        b += 1
        
        optimizer.metricsReporter?.report()
      }
      
      onEpochCompleted?()
      print("----epoch \(i) completed: \(CFAbsoluteTimeGetCurrent() - startTime)s-----")
    }
    
    optimizer.isTraining = false
  }
  
  @discardableResult
  /// Exports the underlying sequential model when available.
  ///
  /// - Parameters:
  ///   - overrite: When `false`, appends a timestamp to avoid overwrite.
  ///   - compress: When `true`, writes compact JSON.
  /// - Returns: URL to exported model, or `nil` when unsupported.
  public func export(overrite: Bool = false, compress: Bool = true) -> URL? {
    if let network = optimizer.trainable as? Sequential {
      return network.export(overrite: overrite, compress: compress)
    }
    
    return nil
  }
  
  private func splitDataset(_ data: [DatasetModel]) -> (data: [Tensor], labels: [Tensor]) {
    var labels: [Tensor] = []
    var input: [Tensor] = []
    
    data.forEach { d in
      labels.append(d.label)
      input.append(d.data)
    }
    
    return (input, labels)
  }
  
  private func trainOn(_ batch: [Tensor],
                       labels: [Tensor],
                       validation: Bool = false,
                       requiresGradients: Bool = true) -> Optimizer.Output {
    optimizer.fit(batch,
                  labels: labels,
                  wrt: nil,
                  lossFunction: lossFunction,
                  validation: validation,
                  requiresGradients: requiresGradients)
  }

}
