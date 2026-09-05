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
  /// The token ID used to pad sequences to a fixed length.
  var padTokenId: Int { get }

  /// The token ID marking the start of a sequence.
  var bosTokenId: Int { get }

  /// IDs of tokens that carry no surface text, such as padding and sequence markers.
  var controlTokenIds: Set<Int> { get }

  /// Encodes an item into its full token sequence.
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
  
  
  /// Exports the dataset's vectorizer to disk and returns the resulting file URL.
  ///
  /// - Parameters:
  ///   - name: Optional filename prefix for the exported file.
  ///   - overrite: When `false`, a timestamp is appended to avoid overwriting.
  ///   - compress: When `true`, the exported file uses compact JSON.
  /// - Returns: URL to the exported file, or `nil` on failure.
  func export(name: String?, overrite: Bool, compress: Bool) -> URL?
  
  /// The number of tokens an item occupies once encoded.
  ///
  /// - Parameters:
  ///   - item: Item to measure.
  ///   - addingBoundaryTokens: Counts the `<bos>`/`<eos>` wrapper when `true`.
  /// - Returns: Token count.
  func tokenCount(for item: String, addingBoundaryTokens: Bool) -> Int
  
  /// Builds an aligned input/label pair for next-token prediction.
  ///
  /// The label is the input advanced by exactly one token, so at timestep `i` the model sees
  /// token `i` and is scored on token `i + 1`. The shift has to happen here, on the token
  /// sequence -- shifting the *text* and re-encoding does not work, because tokenization is
  /// not shift-equivariant: dropping the first character of "the cat" re-encodes as
  /// `["he", "cat", ...]`, leaving every position after the first identical to the input and
  /// training the model to copy rather than predict.
  ///
  /// Wrapping in `<bos>`/`<eos>` makes the first real token a prediction target and teaches
  /// the model to terminate, which is what `RNN.predict` keys off to stop generating.
  ///
  /// - Parameters:
  ///   - item: Item to build a training pair from.
  ///   - sequenceLength: Token count both tensors are padded or truncated to.
  ///   - addingBoundaryTokens: Wraps the item in `<bos>`/`<eos>` when `true`.
  /// - Returns: Input and label tensors, each of depth `sequenceLength`.
  func nextTokenPair(for item: String,
                     sequenceLength: Int,
                     addingBoundaryTokens: Bool) -> (data: Tensor, label: Tensor)
  
  /// The sequence length that fits the longest item in `items` without truncation.
  ///
  /// A run of `n` tokens yields `n - 1` next-token pairs -- the last token is only ever a
  /// label, never an input -- so the returned length is one less than the longest run.
  ///
  /// - Parameters:
  ///   - items: Items the model will train on.
  ///   - cappedAt: Optional upper bound. Longer items are truncated rather than stretching
  ///     every other sample's sequence to match, which costs an LSTM timestep per token.
  ///   - addingBoundaryTokens: Accounts for the `<bos>`/`<eos>` wrapper when `true`.
  /// - Returns: Sequence length to pass to `nextTokenPair(for:sequenceLength:)`.
  func sequenceLength(for items: [String],
                      cappedAt cap: Int?,
                      addingBoundaryTokens: Bool) -> Int
  
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
}

/// A base implementation of `VectorizingDataset` backed by a `Vectorizer` instance.
///
/// Provides default implementations for one-hot encoding, vectorization, decoding,
/// and model export. Subclasses should override `build()` to supply training data.
open class TokenizableDataset: TokenizingDataset {
  /// The concrete tokenizer type used by this dataset.
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

  /// The token ID marking the start of a sequence.
  public var bosTokenId: Int {
    tokenizer.bosTokenId
  }

  /// The token ID used to pad sequences to a fixed length.
  public var padTokenId: Int {
    tokenizer.padTokenId
  }

  /// The number of tokens an item occupies once encoded.
  ///
  /// - Parameters:
  ///   - item: Item to measure.
  ///   - addingBoundaryTokens: Counts the `<bos>`/`<eos>` wrapper when `true`.
  /// - Returns: Token count.
  public func tokenCount(for item: String, addingBoundaryTokens: Bool = true) -> Int {
    tokenizer.encode(item).count + (addingBoundaryTokens ? 2 : 0)
  }

  /// The sequence length that fits the longest item in `items` without truncation.
  ///
  /// A run of `n` tokens yields `n - 1` next-token pairs -- the last token is only ever a
  /// label, never an input -- so the returned length is one less than the longest run.
  ///
  /// - Parameters:
  ///   - items: Items the model will train on.
  ///   - cappedAt: Optional upper bound. Longer items are truncated rather than stretching
  ///     every other sample's sequence to match, which costs an LSTM timestep per token.
  ///   - addingBoundaryTokens: Accounts for the `<bos>`/`<eos>` wrapper when `true`.
  /// - Returns: Sequence length to pass to `nextTokenPair(for:sequenceLength:)`.
  public func sequenceLength(for items: [String],
                             cappedAt cap: Int? = nil,
                             addingBoundaryTokens: Bool = true) -> Int {
    let longest = items
      .map { tokenCount(for: $0, addingBoundaryTokens: addingBoundaryTokens) }
      .max() ?? 0
    let length = max(0, longest - 1)
    return cap.map { min(length, $0) } ?? length
  }

  /// Builds an aligned input/label pair for next-token prediction.
  ///
  /// The label is the input advanced by exactly one token, so at timestep `i` the model sees
  /// token `i` and is scored on token `i + 1`. The shift has to happen here, on the token
  /// sequence -- shifting the *text* and re-encoding does not work, because tokenization is
  /// not shift-equivariant: dropping the first character of "the cat" re-encodes as
  /// `["he", "cat", ...]`, leaving every position after the first identical to the input and
  /// training the model to copy rather than predict.
  ///
  /// Wrapping in `<bos>`/`<eos>` makes the first real token a prediction target and teaches
  /// the model to terminate, which is what `RNN.predict` keys off to stop generating.
  ///
  /// - Parameters:
  ///   - item: Item to build a training pair from.
  ///   - sequenceLength: Token count both tensors are padded or truncated to.
  ///   - addingBoundaryTokens: Wraps the item in `<bos>`/`<eos>` when `true`.
  /// - Returns: Input and label tensors, each of depth `sequenceLength`.
  public func nextTokenPair(for item: String,
                            sequenceLength: Int,
                            addingBoundaryTokens: Bool = true) -> (data: Tensor, label: Tensor) {
    guard sequenceLength > 0 else { return (Tensor(), Tensor()) }

    var ids = tokenizer.encode(item)

    if addingBoundaryTokens {
      ids.insert(tokenizer.bosTokenId, at: 0)
      ids.append(tokenizer.eosTokenId)
    }

    // The pair spans one more token than it emits: input takes all but the last, label all
    // but the first. Truncating to `sequenceLength` alone would cost the final label.
    let span = sequenceLength + 1
    if ids.count > span {
      ids = Array(ids[0..<span])
    }

    guard ids.count >= 2 else { return (Tensor(), Tensor()) }

    let inputIds = padded(Array(ids.dropLast()), to: sequenceLength)
    let labelIds = padded(Array(ids.dropFirst()), to: sequenceLength)

    return (tensor(for: inputIds), tensor(for: labelIds))
  }

  private func padded(_ ids: [Int], to length: Int) -> [Int] {
    guard ids.count < length else { return Array(ids[0..<length]) }
    return ids + [Int](repeating: tokenizer.padTokenId, count: length - ids.count)
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
