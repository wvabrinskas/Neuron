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
  
  /// Applies Mixup augmentation to an input batch and labels.
  ///
  /// - Parameters:
  ///   - input: Input feature tensors.
  ///   - labels: Corresponding label tensors.
  /// - Returns: Mixed inputs, mixed labels, and sampled mix coefficient.
  public func augment(_ input: TensorBatch, labels: TensorBatch) -> AugementedDatasetModel {
    start()
    
    let randomIndices = Array(0..<input.count).randomize()
    
    var mixedInputs: TensorBatch = []
    var mixedLabels: TensorBatch = []
    
    for (i, j) in randomIndices.enumerated() {
      let mixedInput = lambda * input[i] + (1 - lambda) * input[j]
      let mixedLabel = lambda * labels[i] + (1 - lambda) * labels[j]
      
      mixedInputs.append(mixedInput)
      mixedLabels.append(mixedLabel)
    }
    
    return .init(mixed: mixedInputs,
                 mixedLabels: mixedLabels,
                 lambda: lambda)
  }
  
  private func start() {
    lambda = if alpha > 0 {
      BetaDistribution.randomBeta(alpha, beta)
    } else {
      1.0
    }
  }
}

