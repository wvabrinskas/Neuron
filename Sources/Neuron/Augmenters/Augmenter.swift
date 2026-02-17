//
//  Augmenter.swift
//  Neuron
//
//  Created by William Vabrinskas on 2/11/26.
//

public enum Augmenter {
  case mixup(Tensor.Scalar, Tensor.Scalar)
  
  var augmenting: Augmenting {
    switch self {
    case .mixup(let alpha, let beta):
      return Mixup(alpha: alpha, beta: beta)
    }
  }
}

public struct AugementedDatasetModel {
  var mixed: TensorBatch
  var mixedLabels: TensorBatch
  var lambda: Tensor.Scalar
}

public protocol Augmenting {
  /// Applies a data-augmentation transform to a mini-batch.
  ///
  /// - Parameters:
  ///   - inputs: Input feature tensors.
  ///   - labels: Label tensors aligned with `inputs`.
  /// - Returns: Augmented inputs/labels and transform metadata.
  func augment(_ inputs: TensorBatch, labels: TensorBatch) -> AugementedDatasetModel
}
