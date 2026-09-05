//
//  Tokenizing.swift
//  Neuron
//
//  Created by William Vabrinskas on 9/4/26.
//

import Foundation

/// A type alias representing a corpus of text strings used for tokenizer training.
public typealias TokenizerCorpus = [String]

/// A protocol that defines tokenization capabilities with support for encoding, decoding, and exporting trained models.
public protocol Tokenizing: Exportable, Importable {
  var vocabSize: Int { get }
  /// The token ID used to pad sequences out to a fixed length.
  var padTokenId: Int { get }
  /// The token ID marking the start of a sequence.
  var bosTokenId: Int { get }
  /// The token ID marking the end of a generated sequence.
  var eosTokenId: Int { get }
  /// IDs of tokens that carry no surface text, such as padding and sequence markers.
  var controlTokenIds: Set<Int> { get }
  
  /// Trains the tokenizer on the given corpus, building the vocabulary and merge rules.
  ///
  /// - Parameter corpus: An array of text strings used to fit the tokenizer.
  func train(corpus: TokenizerCorpus)

  /// Encodes a text string into a sequence of integer token IDs.
  ///
  /// - Parameter input: The string to encode.
  /// - Returns: A sequence of integer token IDs.
  func encode(_ input: String) -> [Int]

  /// Decodes a sequence of integer token IDs back into a text string.
  ///
  /// - Parameter ids: The integer token IDs to decode.
  /// - Returns: The reconstructed text string.
  func decode(_ ids: [Int]) -> String

  /// Decodes a sequence of integer token IDs back into a text string.
  ///
  /// - Parameters:
  ///   - ids: The integer token IDs to decode.
  ///   - skipControlTokens: When `true`, tokens with no surface text are dropped instead of
  ///     being rendered as their literal names.
  /// - Returns: The reconstructed text string.
  func decode(_ ids: [Int], skipControlTokens: Bool) -> String
  
}
