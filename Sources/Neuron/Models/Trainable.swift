//
//  File.swift
//  
//
//  Created by William Vabrinskas on 3/15/22.
//

import Foundation

public protocol Trainable {
  associatedtype TrainableDatasetType
  typealias InputData = (training: [TrainableDatasetType], validation: [TrainableDatasetType])
  
  func train(dataset: InputData,
             epochCompleted: ((_ epoch: Int, _ metrics: [Metric: Float]) -> ())?,
             complete: ((_ metrics: [Metric: Float]) -> ())?)
  
  func trainOn(_ batch: [TrainableDatasetType]) -> Float
  func validateOn(_ batch: [TrainableDatasetType]) -> Float
}

public extension Trainable {
  func validateOn(_ batch: [TrainingData]) -> Float {
    print("Using default implementation of: ", #function)
    //no op
    return 0
  }
  
  func trainOn(_ batch: [TrainingData]) -> Float {
    print("Using default implementation of: ", #function)
    //no op
    return 0
  }
}
