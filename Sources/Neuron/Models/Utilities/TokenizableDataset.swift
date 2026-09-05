//
//  TokenizableDataset.swift
//  Neuron
//
//  Created by William Vabrinskas on 9/4/26.
//

import Foundation

/// A tuple containing training and validation dataset model arrays.
public typealias TokenizingDatasetData = (training: [DatasetModel], val: [DatasetModel])
/// A protocol for datasets that support vectorization of their items.
///
/// Conforming types provide a vectorizer, vocabulary size, and methods
/// for encoding items as one-hot tensors or index-based tensors.
public protocol TokenizingDataset {
  associatedtype Tokenizer: Tokenizing
  
  /// The element type this dataset vectorizes, currently fixed to `String`.
  typealias Item = String
  /// The vectorizer used to encode and decode dataset items.
  var tokenizer: Tokenizer { get }
  /// The total number of unique tokens in the vocabulary.
  var vocabSize: Int { get }
  /// The token ID marking the end of a generated sequence.
  var eosTokenId: Int { get }
  /// IDs of tokens that carry no surface text, such as padding and sequence markers.
  var controlTokenIds: Set<Int> { get }

  ///
  /// - Parameter items: Items to vectorize.
  /// - Returns: Tensor containing vectorized token IDs.
  func tokenize(_ items: Item) -> Tensor
  /// Encodes an item and pads or truncates it to a fixed token count.
  ///
  /// - Parameters:
  ///   - item: Item to tokenize.
  ///   - length: Exact number of tokens the returned tensor should carry.
  ///   - appendingEnd: When `true`, an end-of-sequence token is placed after the content.
  /// - Returns: Tensor of `length` token IDs, one per depth slice.
  func tokenize(_ item: Item, paddedTo length: Int, appendingEnd: Bool) -> Tensor
  /// Decodes model output tensor values back into dataset items.
  ///
  /// - Parameters:
  ///   - data: Tensor to decode.
  ///   - oneHot: Whether `data` is one-hot encoded.
  /// - Returns: Decoded item sequence.
  func item(for data: Tensor) -> Item
  /// Builds training and validation datasets for RNN training.
  ///
  /// - Returns: Tuple containing training and validation datasets.
  func build() async -> TokenizingDatasetData
  
  /// Builds a dataset instance by loading a vectorizer from a file URL.
  ///
  /// - Parameter url: File URL from which to import vectorizer state.
  /// - Returns: A new dataset instance initialized with the imported vectorizer.
  static func build(url: URL) -> Self

  @_spi(Visualizer)
  /// Builds a dataset instance by loading a vectorizer from raw data bytes.
  ///
  /// - Parameter data: Raw data from which to import vectorizer state.
  /// - Returns: A new dataset instance initialized with the imported vectorizer.
  static func build(data: Data) -> Self

  /// Exports the dataset's vectorizer to disk and returns the resulting file URL.
  ///
  /// - Parameters:
  ///   - name: Optional filename prefix for the exported file.
  ///   - overrite: When `false`, a timestamp is appended to avoid overwriting.
  ///   - compress: When `true`, the exported file uses compact JSON.
  /// - Returns: URL to the exported file, or `nil` on failure.
  func export(name: String?, overrite: Bool, compress: Bool) -> URL?
}

/// A base implementation of `VectorizingDataset` backed by a `Vectorizer` instance.
///
/// Provides default implementations for one-hot encoding, vectorization, decoding,
/// and model export. Subclasses should override `build()` to supply training data.
open class TokenizableDataset: TokenizingDataset {
  public typealias Tokenizer = BPETokenizer

  /// The vectorizer used to encode and decode dataset items.
  public let tokenizer: Tokenizer

  /// The number of unique tokens in the vocabulary.
  ///
  /// Reflects the size of the vectorizer's internal vector mapping.
  public var vocabSize: Int {
    tokenizer.vocabSize
  }

  /// The token ID marking the end of a generated sequence.
  public var eosTokenId: Int {
    tokenizer.eosTokenId
  }

  /// IDs of tokens that carry no surface text, such as padding and sequence markers.
  public var controlTokenIds: Set<Int> {
    tokenizer.controlTokenIds
  }

  /// Creates a dataset backed by the given vectorizer.
  ///
  /// - Parameter vectorizer: Vectorizer used to encode and decode dataset items.
  public required init(tokenizer: Tokenizer) {
    self.tokenizer = tokenizer
  }

/// Creates a dataset instance by importing a vectorizer from a file URL.
///
/// - Parameter url: The file URL from which to import the vectorizer.
/// - Returns: A new instance initialized with the imported vectorizer.
  public static func build(url: URL) -> Self {
    Self.init(tokenizer: Tokenizer.import(url))
  }

  @_spi(Visualizer)
/// Creates a dataset instance by importing a vectorizer from raw data.
///
/// - Parameter data: The raw data from which to import the vectorizer.
/// - Returns: A new instance initialized with the imported vectorizer.
  public static func build(data: Data) -> Self {
    return Self.init(tokenizer: Tokenizer.import(data))
  }
  
/// Exports the vectorizer to a file and returns the resulting file URL.
///
/// - Parameter name: An optional name for the exported file.
/// - Parameter overrite: Whether to overwrite an existing file at the destination.
/// - Parameter compress: Whether to compress the exported file.
/// - Returns: The URL of the exported file, or `nil` if export failed.
  public func export(name: String?, overrite: Bool, compress: Bool) -> URL?  {
    tokenizer.export(name: name, overrite: overrite, compress: compress)
  }
  
  /// Converts an item into integer token IDs wrapped in a tensor.
  ///
  /// Token IDs are laid out one-per-depth-slice (`rows: 1, columns: 1, depth: tokenCount`),
  /// which is the layout `Embedding` expects: it reads a single index from each depth slice.
  /// Packing the IDs along `columns` instead would leave the tensor at `depth: 1` and the
  /// network would only ever see the first token.
  ///
  /// - Parameter item: Item to tokenize.
  /// - Returns: Tensor containing one token ID per depth slice.
  public func tokenize(_ item: String) -> Tensor {
    tensor(for: tokenizer.encode(item))
  }

  /// Converts an item into token IDs padded (or truncated) to a fixed token count.
  ///
  /// Recurrent models compile against a single sequence length, so every sample has to
  /// produce the same token count. BPE merges make raw encodings ragged, hence the padding.
  ///
  /// - Parameters:
  ///   - item: Item to tokenize.
  ///   - length: Exact number of tokens the returned tensor should carry.
  ///   - appendingEnd: When `true`, an end-of-sequence token is placed after the content
  ///     (truncating it by one if necessary) so a model trained on these sequences learns to
  ///     emit a terminator. Generation cannot stop on its own otherwise.
  /// - Returns: Tensor containing `length` token IDs, one per depth slice.
  public func tokenize(_ item: String, paddedTo length: Int, appendingEnd: Bool = false) -> Tensor {
    guard length > 0 else { return Tensor() }

    var ids = tokenizer.encode(item)
    let contentLength = appendingEnd ? length - 1 : length

    if ids.count > contentLength {
      ids = Array(ids[0..<max(0, contentLength)])
    }

    if appendingEnd {
      ids.append(tokenizer.eosTokenId)
    }

    if ids.count < length {
      ids.append(contentsOf: [Int](repeating: tokenizer.padTokenId,
                                   count: length - ids.count))
    }

    return tensor(for: ids)
  }

  private func tensor(for ids: [Int]) -> Tensor {
    guard ids.isEmpty == false else { return Tensor() }
    return Tensor(ids.map { [[Tensor.Scalar($0)]] })
  }
  
  /// Decodes model output back into vector items.
  ///
  /// - Parameters:
  ///   - data: Tensor to decode.
  ///   - oneHot: Whether `data` uses one-hot encoding.
  /// - Returns: Decoded vector items.
  public func item(for data: Tensor) -> String {
    let intArray = data.storage.map { Int($0) }
    // Control tokens are skipped: they exist to shape the tensor, and rendering a padded
    // sequence as "<pad><pad><pad>" is never what a caller wants back from a model output.
    return tokenizer.decode(intArray, skipControlTokens: true)
  }
  
  /// Builds dataset content for RNN training.
  ///
  /// Subclasses should override with concrete dataset construction.
  ///
  /// - Returns: Empty training/validation datasets by default.
  open func build() async -> TokenizingDatasetData {
    ([], [])
  }
}
