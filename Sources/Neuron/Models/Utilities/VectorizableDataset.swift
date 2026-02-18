//
//  VectorizableDataset.swift
//  Neuron
//
//  Created by William Vabrinskas on 1/9/26.
//

import Foundation


public typealias VectorizingDatasetData = (training: [DatasetModel], val: [DatasetModel])
public protocol VectorizingDataset {
  associatedtype Item: VectorizableItem
  
  var vectorizer: Vectorizer<Item> { get }
  
  var vocabSize: Int { get }
  /// One-hot encodes dataset items.
  ///
  /// - Parameter items: Items to encode.
  /// - Returns: One-hot tensor representation.
  func oneHot(_ items: [Item]) -> Tensor
  /// Vectorizes dataset items into index-based representation.
  ///
  /// - Parameter items: Items to vectorize.
  /// - Returns: Tensor containing vectorized token IDs.
  func vectorize(_ items: [Item]) -> Tensor
  /// Decodes model output tensor values back into dataset items.
  ///
  /// - Parameters:
  ///   - data: Tensor to decode.
  ///   - oneHot: Whether `data` is one-hot encoded.
  /// - Returns: Decoded item sequence.
  func getWord(for data: Tensor, oneHot: Bool) -> [Item]
  /// Builds training and validation datasets for RNN training.
  ///
  /// - Returns: Tuple containing training and validation datasets.
  func build() async -> VectorizingDatasetData
  
  static func build(url: URL) -> Self
  
  @_spi(Visualizer)
  static func build(data: Data) -> Self
  
  func export(name: String?, overrite: Bool, compress: Bool) -> URL?
}

open class VectorizableDataset<VectorItem: VectorizableItem>: VectorizingDataset {

  public let vectorizer: Vectorizer<VectorItem>

  public var vocabSize: Int = 0

  public required init(vectorizer: Vectorizer<VectorItem> = .init()) {
    self.vectorizer = vectorizer
    self.vocabSize = vectorizer.vector.count
  }

  public static func build(url: URL) -> Self {
    Self.init(vectorizer: Vectorizer<VectorItem>.import(url))
  }

  @_spi(Visualizer)
  public static func build(data: Data) -> Self {
    return Self.init(vectorizer: Vectorizer<VectorItem>.import(data))
  }
  
  public func export(name: String?, overrite: Bool, compress: Bool) -> URL?  {
    vectorizer.export(name: name, overrite: overrite, compress: compress)
  }
  
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
  open func build() async -> VectorizingDatasetData {
    ([], [])
  }
}
