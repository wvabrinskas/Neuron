//
//  Mixup.swift
//
//

import Foundation
import NumSwift

public final class Mixup: Augmenting {
  private let alpha: Tensor.Scalar
  
  public init(alpha: Tensor.Scalar = 0.2) {
    self.alpha = alpha
  }
  
  public func augment(_ input: TensorBatch, labels: TensorBatch) -> AugementedDatasetModel {
    let randomIndicies = Array(0..<input.count).randomize()
    
    let lambda: Tensor.Scalar = if alpha > 0 {
      BetaDistribution.randomBeta(alpha: alpha)
    } else {
      1.0
    }
    
    var mixedLabels: TensorBatch = []
    
    let mixed = zip(input, randomIndicies).map { x, i in
      mixedLabels.append(labels[i])
      return lambda * x + (1 - lambda) * input[i]
    }
    
    return .init(mixed: mixed,
                 mixedLabels: mixedLabels,
                 labels: labels,
                 lambda: lambda)
  }
}

