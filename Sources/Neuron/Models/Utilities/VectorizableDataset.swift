//
//  VectorizableDataset.swift
//  Neuron
//
//  Created by William Vabrinskas on 1/9/26.
//

open class VectorizableDataset<VectorItem: Hashable>: RNNSupportedDataset {

  public let vectorizer: Vectorizer<VectorItem> = .init()
  
  public var vocabSize: Int = 0
  
  /// One-hot encodes the provided items using the internal vectorizer.
  ///
  /// - Parameter items: Items to encode.
  /// - Returns: One-hot tensor representation.
  public func oneHot(_ items: [VectorItem]) -> Tensor {
    vectorizer.oneHot(items)
  }
  
  /// Converts items into integer token IDs wrapped in a tensor.
  ///
  /// - Parameter items: Items to vectorize.
  /// - Returns: Tensor containing token IDs per depth slice.
  public func vectorize(_ items: [VectorItem]) -> Tensor {
    Tensor(items.map { [[Tensor.Scalar(vectorizer.vector[$0, default: 0])]] })
  }
  
  /// Decodes model output back into vector items.
  ///
  /// - Parameters:
  ///   - data: Tensor to decode.
  ///   - oneHot: Whether `data` uses one-hot encoding.
  /// - Returns: Decoded vector items.
  public func getWord(for data: Tensor, oneHot: Bool) -> [VectorItem] {
    if oneHot == false {
      let intArray = data.storage.map { Int($0) }
      return vectorizer.unvectorize(intArray)
    } else {
      return vectorizer.unvectorizeOneHot(data)
    }
  }
  
  /// Builds dataset content for RNN training.
  ///
  /// Subclasses should override with concrete dataset construction.
  ///
  /// - Returns: Empty training/validation datasets by default.
  open func build() async -> RNNSupportedDatasetData {
    ([], [])
  }
}
