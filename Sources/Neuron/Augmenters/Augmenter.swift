//
//  Augmenter.swift
//  Neuron
//
//  Created by William Vabrinskas on 2/11/26.
//

/// An enumeration of supported data augmentation strategies.
///
/// Each case encapsulates the configuration parameters required
/// for a specific augmentation technique.
///
/// - Case mixup: Applies Mixup augmentation using the specified alpha and beta
///   distribution parameters.
public enum Augmenter {
  case mixup(Tensor.Scalar, Tensor.Scalar)
  
  var augmenting: Augmenting {
    switch self {
    case .mixup(let alpha, let beta):
      return Mixup(alpha: alpha, beta: beta)
    }
  }
}

/// A model representing the output of an augmented dataset transformation.
///
/// Contains the blended input tensors, blended label tensors, and the
/// interpolation coefficient produced during augmentation.
public struct AugementedDatasetModel {
  var mixed: TensorBatch
  var mixedLabels: TensorBatch
  var lambda: Tensor.Scalar
}

/// A protocol that defines a data augmentation transform applicable to mini-batches.
public protocol Augmenting {
  /// Applies a data-augmentation transform to a mini-batch.
  ///
  /// - Parameters:
  ///   - inputs: Input feature tensors.
  ///   - labels: Label tensors aligned with `inputs`.
  /// - Returns: Augmented inputs/labels and transform metadata.
  func augment(_ inputs: TensorBatch, labels: TensorBatch) -> AugementedDatasetModel
}
