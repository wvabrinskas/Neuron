//
//  that.swift
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
  
}
