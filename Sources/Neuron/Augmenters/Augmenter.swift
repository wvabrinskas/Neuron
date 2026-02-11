//
//  Augmenter.swift
//  Neuron
//
//  Created by William Vabrinskas on 2/11/26.
//

public enum Augmenter {
  case mixup(Tensor.Scalar)
  
  var augmenting: Augmenting {
    switch self {
    case .mixup(let alpha):
      return Mixup(alpha: alpha)
    }
  }
}

public struct AugementedDatasetModel {
  var mixed: TensorBatch
  var mixedLabels: TensorBatch
  var lambda: Tensor.Scalar
}

public protocol Augmenting {
  func augment(_ inputs: TensorBatch, labels: TensorBatch) -> AugementedDatasetModel
  func adjustForAugment(_ a: Tensor, _ b: Tensor) -> Tensor
}
