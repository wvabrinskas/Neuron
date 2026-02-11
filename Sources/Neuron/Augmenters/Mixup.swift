//
//  Mixup.swift
//
//

import Foundation
import NumSwift

public final class Mixup: Augmenting {
  private let alpha: Tensor.Scalar
  private let beta: Tensor.Scalar
  
  private var lambda: Tensor.Scalar = 0
  
  
  /// Initializer that creates a distribution centered around 0.5 by default
  /// - Parameters:
  ///   - alpha: alpha paramater for BetaDistribution. default = 0.2
  ///   - beta: beta paramater for BetaDistribution. default = 0.2
  public init(alpha: Tensor.Scalar = 0.2,
              beta: Tensor.Scalar = 0.2) {
    self.alpha = alpha
    self.beta = beta
  }
  
  public func augment(_ input: TensorBatch, labels: TensorBatch) -> AugementedDatasetModel {
    start()
    
    let randomIndicies = Array(0..<input.count).randomize()
  
    var mixedLabels: TensorBatch = []
    
    let mixed = zip(input, randomIndicies).map { x, i in
      mixedLabels.append(labels[i])
      return lambda * x + (1 - lambda) * input[i]
    }
    
    return .init(mixed: mixed,
                 mixedLabels: mixedLabels,
                 lambda: lambda)
  }
  
  public func adjustForAugment(_ a: Tensor, _ b: Tensor) -> Tensor {
    (lambda * a + (1 - lambda) * b)
  }
  
  private func start() {
    lambda = if alpha > 0 {
      BetaDistribution.randomBeta(alpha, beta)
    } else {
      1.0
    }
  }
}

