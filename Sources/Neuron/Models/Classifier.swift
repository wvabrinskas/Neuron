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
  private let threadWorkers: Int
  private let epochs: Int
  private let log: Bool
  private let lossFunction = LossFunction.crossEntropySoftmax
  private let accuracyThreshold: Float
  private let killOnAccuracy: Bool
  
  public var onAccuracyReached: (() -> ())? = nil
  public var onEpochCompleted: (() -> ())? = nil
  
  var optimNetwork: Optimizer
  
  public init(optimizer: Optimizer,
              epochs: Int = 100,
              batchSize: Int,
              accuracyThreshold: Float = 0.8,
              killOnAccuracy: Bool = true,
              threadWorkers: Int = 16,
              log: Bool = false) {
    self.batchSize = batchSize
    self.accuracyThreshold = accuracyThreshold
    self.threadWorkers = threadWorkers
    self.optimNetwork = optimizer
    self.epochs = epochs
    self.killOnAccuracy = killOnAccuracy
    self.log = log
  }
  
  public func feed(_ data: [Tensor]) -> [Tensor] {
    optimNetwork(data)
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
    
    optimNetwork.isTraining = true

    for i in 0..<epochs {
      let startTime = Date().timeIntervalSince1970

      var b = 0
      
      if let randomValBatch = validationBatches.randomElement() {
        
        let result = trainOn(randomValBatch.data,
                             labels: randomValBatch.labels,
                             validation: true,
                             requiresGradients: false)
        let loss = result.loss
        optimNetwork.metricsReporter?.update(metric: .valLoss, value: loss)
        optimNetwork.metricsReporter?.update(metric: .valAccuracy, value: result.accuracy)

        print("val_complete - ", "loss: ", loss, "accuracy: ", result.accuracy)
        
        if result.accuracy >= (accuracyThreshold * 100) {
          self.onAccuracyReached?()
          if killOnAccuracy {
            return
          }
        }
      }
            
      for batch in trainingBatches {
        optimNetwork.zeroGradients()

        let result = trainOn(batch.data, labels: batch.labels)
        let weightGradients = result.gradients
        let loss = result.loss
        
        optimNetwork.metricsReporter?.update(metric: .loss, value: loss)
        optimNetwork.metricsReporter?.update(metric: .accuracy, value: result.accuracy)

        let batchesCompletePercent = (round((Double(b) / Double(batches.count)) * 10000) / 10000) * 100
        
        if log {
          print("complete :", "\(b) / \(batches.count) -> \(batchesCompletePercent)%")
        }
        
        optimNetwork.apply(weightGradients)
        optimNetwork.step()
        b += 1
        
        optimNetwork.metricsReporter?.report()
      }
      
      onEpochCompleted?()
      print("----epoch \(i) completed: \(Date().timeIntervalSince1970 - startTime)s-----")
    }
    
    optimNetwork.isTraining = false
  }
  
  @discardableResult
  public func export(overrite: Bool = false) -> URL? {
    if let network = optimNetwork.trainable as? Sequential {
      let additional = overrite == false ? "-\(Date().timeIntervalSince1970)" : ""
      
      let dUrl = ExportHelper.getModel(filename: "classifier\(additional)", model: network)
      
      return dUrl
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
    optimNetwork.fit(batch,
                     labels: labels,
                     lossFunction: lossFunction,
                     validation: validation,
                     requiresGradients: requiresGradients)
  }

}
