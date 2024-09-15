//
//  Accuracy.swift
//  Neuron
//
//  Created by William Vabrinskas on 9/14/24.
//

import NumSwift
import Foundation


/// Used for a model to send desired accuracy information to a model. 
public struct AccuracyThreshold: Hashable {
  let value: Tensor.Scalar
  let averageCount: Int
  
  
  /// Default initializer
  /// - Parameters:
  ///   - value: The desired accuracy, a value between 0 and 1.
  ///   - averageCount: The number of samples to average over.
  public init(value: Tensor.Scalar, averageCount: Int) {
    self.value = Swift.max(Swift.min(1, value), 0)
    self.averageCount = averageCount
  }
}

final class MovingMean {
  typealias Value = [Tensor.Scalar]
  let count: Int
  var value: Value
  
  var mean: Tensor.Scalar {
    value.mean
  }
  
  init(count: Int, value: Value? = nil) {
    self.count = count
    self.value = value ?? [Tensor.Scalar](repeating: 0, count: count)
  }
  
  @inlinable
  func append(_ scalar: Tensor.Scalar) {
    if value.count >= count {
      value.removeFirst()
    }
    
    value.append(scalar)
  }
  
  func reset() {
    value.removeAll(keepingCapacity: true)
  }
  
}

final class AccuracyMonitor {
  let threshold: AccuracyThreshold
  let movingMean: MovingMean
  
  init(threshold: AccuracyThreshold) {
    self.threshold = threshold
    self.movingMean = .init(count: threshold.averageCount)
  }
    
  func append(_ scalar: Tensor.Scalar) {
    movingMean.append(scalar)
  }
  
  func mean() -> Tensor.Scalar {
    movingMean.mean
  }
  
  func isAboveThreshold() -> Bool {
    mean() > threshold.value
  }
}
