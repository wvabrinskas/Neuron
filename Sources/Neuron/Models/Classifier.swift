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
  
  public func feed(_ data: [Tensor]) -> [Tensor] {
    optimizer(data)
  }
  
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
