//
//  VectorizableDataset.swift
//  Neuron
//
//  Created by William Vabrinskas on 1/9/26.
//

import Foundation


/// A tuple containing training and validation dataset model arrays.
public typealias VectorizingDatasetData = (training: [DatasetModel], val: [DatasetModel])
/// A protocol for datasets that support vectorization of their items.
///
/// Conforming types provide a vectorizer, vocabulary size, and methods
/// for encoding items as one-hot tensors or index-based tensors.
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

/// The vectorizer used to encode and decode dataset items.
  public let vectorizer: Vectorizer<VectorItem>

/// The number of unique tokens in the vocabulary.
///
/// Reflects the size of the vectorizer's internal vector mapping.
  public var vocabSize: Int = 0

  public required init(vectorizer: Vectorizer<VectorItem> = .init()) {
    self.vectorizer = vectorizer
    self.vocabSize = vectorizer.vector.count
  }

/// Creates a dataset instance by importing a vectorizer from a file URL.
///
/// - Parameter url: The file URL from which to import the vectorizer.
/// - Returns: A new instance initialized with the imported vectorizer.
  public static func build(url: URL) -> Self {
    Self.init(vectorizer: Vectorizer<VectorItem>.import(url))
  }

  @_spi(Visualizer)
/// Creates a dataset instance by importing a vectorizer from raw data.
///
/// - Parameter data: The raw data from which to import the vectorizer.
/// - Returns: A new instance initialized with the imported vectorizer.
  public static func build(data: Data) -> Self {
    return Self.init(vectorizer: Vectorizer<VectorItem>.import(data))
  }
  
/// Exports the vectorizer to a file and returns the resulting file URL.
///
/// - Parameter name: An optional name for the exported file.
/// - Parameter overrite: Whether to overwrite an existing file at the destination.
/// - Parameter compress: Whether to compress the exported file.
/// - Returns: The URL of the exported file, or `nil` if export failed.
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
