//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/3/22.
//

import Foundation
import NumSwift

/// High-level classifier model for supervised learning tasks
/// Handles training loop, validation, and accuracy monitoring
/// Provides convenient interface for training neural networks on classification tasks
public class Classifier {
  /// Number of samples per training batch
  private var batchSize: Int
  /// Number of training epochs to run
  private let epochs: Int
  /// Whether to log training progress
  private let log: Bool
  /// Loss function used for training
  private let lossFunction: LossFunction
  /// Accuracy threshold configuration for early stopping
  private let accuracyThreshold: AccuracyThreshold
  /// Whether to stop training when accuracy threshold is reached
  private let killOnAccuracy: Bool
  /// Monitors accuracy for early stopping decisions
  private let accuracyMonitor: AccuracyMonitor
  
  /// Callback executed when accuracy threshold is reached
  public var onAccuracyReached: (() -> ())? = nil
  /// Callback executed after each epoch completes
  public var onEpochCompleted: (() -> ())? = nil
  
  /// The optimizer used for training the model
  public private(set) var optimizer: Optimizer
  
  /// Initializes a classifier with specified training parameters
  /// - Parameters:
  ///   - optimizer: The optimizer to use for training
  ///   - epochs: Number of training epochs. Default: 100
  ///   - batchSize: Number of samples per batch
  ///   - accuracyThreshold: Accuracy threshold for early stopping. Default: 80% over 5 epochs
  ///   - killOnAccuracy: Whether to stop training when threshold is reached. Default: true
  ///   - log: Whether to log training progress. Default: false
  ///   - lossFunction: Loss function for training. Default: .crossEntropySoftmax
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
  
  /// Feeds data through the model for inference
  /// - Parameter data: Input tensors to process
  /// - Returns: Model predictions
  public func feed(_ data: [Tensor]) -> [Tensor] {
    optimizer(data)
  }
  
  /// Trains the classifier on the provided dataset
  /// Runs for the specified number of epochs with validation monitoring
  /// - Parameters:
  ///   - data: Training dataset
  ///   - validation: Validation dataset for monitoring generalization
  public func fit(_ data: [DatasetModel], _ validation: [DatasetModel]) {
    let batches = data.batched(into: batchSize)
    let valBatches = validation.batched(into: batchSize)

    //epochs
    var trainingBatches: [(data: [Tensor], labels: [Tensor])] = []
    batches.forEach { b in
      let trainingSet = splitDataset(b)
      trainingBatches.append(trainingSet)
    }

    var validationBatches: [(data: [Tensor], labels: [Tensor])] = []
    valBatches.forEach { b in
      let valSet = splitDataset(b)
      validationBatches.append(valSet)
    }
    
    optimizer.isTraining = true

    for i in 0..<epochs {
      let startTime = Date().timeIntervalSince1970

      var b = 0
      
      if let randomValBatch = validationBatches.randomElement() {
        
        let result = trainOn(randomValBatch.data,
                             labels: randomValBatch.labels,
                             validation: true,
                             requiresGradients: false)
        let loss = result.loss
        optimizer.metricsReporter?.update(metric: .valLoss, value: loss)
        optimizer.metricsReporter?.update(metric: .valAccuracy, value: result.accuracy)

        print("val_complete - ", "loss: ", loss, "accuracy: ", result.accuracy)
        
        accuracyMonitor.append(result.accuracy / 100.0)
        
        if accuracyMonitor.isAboveThreshold() {
          self.onAccuracyReached?()
          if killOnAccuracy {
            return
          }
        }
      }
            
      for batch in trainingBatches {
        optimizer.zeroGradients()

        let result = trainOn(batch.data, labels: batch.labels)
        let weightGradients = result.gradients
        let loss = result.loss
        
        optimizer.metricsReporter?.update(metric: .loss, value: loss)
        optimizer.metricsReporter?.update(metric: .accuracy, value: result.accuracy)

        let batchesCompletePercent = (round((Double(b) / Double(batches.count)) * 10000) / 10000) * 100
        
        if log {
          print("complete :", "\(b) / \(batches.count) -> \(batchesCompletePercent)%")
        }
        
        optimizer.apply(weightGradients)
        optimizer.step()
        b += 1
        
        optimizer.metricsReporter?.report()
      }
      
      onEpochCompleted?()
      print("----epoch \(i) completed: \(Date().timeIntervalSince1970 - startTime)s-----")
    }
    
    optimizer.isTraining = false
  }
  
  /// Exports the trained model to a file
  /// - Parameters:
  ///   - overrite: Whether to overwrite existing files. Default: false
  ///   - compress: Whether to compress the exported model. Default: true
  /// - Returns: URL of the exported model file, or nil if export fails
  @discardableResult
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
                     lossFunction: lossFunction,
                     validation: validation,
                     requiresGradients: requiresGradients)
  }

}
