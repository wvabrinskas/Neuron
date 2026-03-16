//
//  Tokenizing.swift
//  Neuron
//
//  Created by William Vabrinskas on 2/18/26.
//

import NumSwift
import Foundation

/// A type alias representing a corpus of text strings used for tokenizer training.
public typealias TokenizerCorpus = [String]

/// A protocol that defines tokenization capabilities with support for encoding, decoding, and exporting trained models.
public protocol Tokenizing: Exportable {
  func train(corpus: TokenizerCorpus)
  func encode(_ input: String) -> [Int]
  func decode(_ ids: [Int]) -> String
}

open class BaseTokenizer: Tokenizing {
  private var vocab: [String: Int] = [:]
  private var reverseVocab: [Int: String] = [:]
  
  private let vectorizer = Vectorizer()
  
  private var corpus: TokenizerCorpus = []
  private var nextId: Int = 0
  
  private let wordEnding: String = "</w>"
  private let wordUnknown: String = "<unk>"
  private let targetVocabSize: Int
  private var mergeRules: [TokenPair] = []

  private struct TokenPair: Hashable, Codable {
    var tokenA: String
    var tokenB: String
    
    func join() -> String {
      tokenA + tokenB
    }
  }
  
  private lazy var specialTokens: [String] = [
      "<pad>",
      wordUnknown,
      "<bos>",
      "<eos>",
      wordEnding
  ]
  
  /// Coding keys used for encoding and decoding the tokenizer's properties.
  public enum CodingKeys: String, CodingKey {
    case mergeRules
    case vocab
    case inverseVocab
    case targetVocabSize
  }

  /// Initializes a new tokenizer with the specified target vocabulary size.
  ///
  /// - Parameter targetVocabSize: The desired number of tokens in the trained vocabulary.
  public init(targetVocabSize: Int) {
    self.targetVocabSize = targetVocabSize
  }

  /// Initializes a tokenizer by decoding it from the given decoder.
  ///
  /// - Parameter decoder: The decoder to read data from.
  /// - Throws: An error if any required values are missing or cannot be decoded.
  required public init(from decoder: any Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.mergeRules = try container.decode([TokenPair].self, forKey: .mergeRules)
    self.vocab = try container.decode([String: Int].self, forKey: .vocab)
    self.targetVocabSize = try container.decode(Int.self, forKey: .targetVocabSize)
    
    self.reverseVocab = Dictionary(uniqueKeysWithValues: vocab.map( { ($1, $0) }))
  }
  
  /// Encodes the tokenizer's state into the given encoder.
  ///
  /// - Parameter encoder: The encoder to write data to.
  /// - Throws: An error if any values fail to encode.
  public func encode(to encoder: any Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(mergeRules, forKey: .mergeRules)
    try container.encode(vocab, forKey: .vocab)
    try container.encode(targetVocabSize, forKey: .targetVocabSize)
  }
  
  open func train(corpus: TokenizerCorpus) {
    let flatCorpus = corpus.joined(separator: " ")
    
    vectorizer.vectorize(specialTokens)
    vectorizer.vectorize(flatCorpus.characters)
    vocab = vectorizer.vector
    reverseVocab = vectorizer.inverseVector
    
    nextId = vectorizer.lastKey + 1
    
    var wordFrequency: [String: Int] = [:]
    let words = flatCorpus.components(separatedBy: " ")
    
    for word in words {
      let characters = word.map { String($0) }.joined(separator: " ") + " " + wordEnding
      wordFrequency[characters, default: 0] += 1
    }
    
    while vocab.count < targetVocabSize {
      var pairCounts: [TokenPair: Int] = [:]
      
      for (word, freq) in wordFrequency {
        let tokens = word.components(separatedBy: " ")
        for (tokenA, tokenB) in zip(tokens, tokens.dropFirst()) {
          let pair = TokenPair(tokenA: tokenA, tokenB: tokenB)
          pairCounts[pair, default: 0] += freq
        }
      }
      
      guard pairCounts.isEmpty == false,
         let bestPair = pairCounts.sorted(by: {
           if $0.value != $1.value { return $0.value > $1.value }
           if $0.key.tokenA != $1.key.tokenA { return $0.key.tokenA < $1.key.tokenA }
           return $0.key.tokenB < $1.key.tokenB
         }).first?.key else {
        break
      }
      
      let newToken = bestPair.join()
      vocab[newToken] = nextId
      reverseVocab[nextId] = newToken
      
      mergeRules.append(bestPair)
      nextId += 1
      
      wordFrequency = applyMerge(wordFreqs: wordFrequency,
                                 pair: bestPair)
    }
  }
  
  open func encode(_ text: String) -> [Int] {
    
    var tokenIds: [Int] = []
    let words = text.components(separatedBy: " ")
    
    for word in words {
      // Split word into individual characters, append end-of-word marker
      var tokens = word.map { String($0) }
      tokens.append(wordEnding)
      
      // Apply merge rules IN ORDER
      for rule in mergeRules {
        var newTokens: [String] = []
        var i = 0
        while i < tokens.count {
          if i < tokens.count - 1 &&
              tokens[i] == rule.tokenA &&
              tokens[i + 1] == rule.tokenB {
            newTokens.append(rule.tokenA + rule.tokenB)
            i += 2
          } else {
            newTokens.append(tokens[i])
            i += 1
          }
        }
        tokens = newTokens
      }
      
      // Map each token to its ID
      for token in tokens {
        if let id = vocab[token] {
          tokenIds.append(id)
        } else {
          if let id = vocab[wordUnknown] { // fallback for unknown tokens
            tokenIds.append(id)
          }
        }
      }
    }
    
    return tokenIds
  }
  
  open func decode(_ tokenIds: [Int]) -> String {
    // Invert the vocab dictionary
    let idToToken = reverseVocab
    
    // Map IDs back to token strings
    let tokens = tokenIds.compactMap { idToToken[$0] }
    
    // Join and clean up end-of-word markers
    return tokens
      .joined()
      .replacingOccurrences(of: wordEnding, with: " ")
      .trimmingCharacters(in: .whitespaces)
  }
  
  
  @discardableResult
  /// Exports the trainable as a `.stkns` file.
  ///
  /// - Parameters:
  ///   - name: Optional filename prefix.
  ///   - overrite: When `false`, appends a timestamp to avoid overwrite.
  ///   - compress: When `true`, emits compact JSON.
  /// - Returns: URL to the exported model file, or `nil` on write failure.
  public func export(name: String?, overrite: Bool, compress: Bool) -> URL? {
    let additional = overrite == false ? "-\(Date().timeIntervalSince1970)" : ""
    
    let filename = (name ?? "tokens") + additional
    
    let dUrl = ExportHelper.getTokens(filename: filename, compress: compress, model: self)
    
    return dUrl
  }
    
  private func applyMerge(
    wordFreqs: [String: Int],
    pair: TokenPair
  ) -> [String: Int] {
    
    let merged = pair.join()
    var newWordFreqs: [String: Int] = [:]
    
    for (wordKey, freq) in wordFreqs {
      // Split the string key back into individual tokens
      let tokens = wordKey.components(separatedBy: " ")
      
      var newTokens: [String] = []
      var i = 0
      while i < tokens.count {
        if i < tokens.count - 1 &&
            tokens[i] == pair.tokenA &&
            tokens[i + 1] == pair.tokenB {
          newTokens.append(merged)
          i += 2            // skip both, we merged them
        } else {
          newTokens.append(tokens[i])
          i += 1
        }
      }
      
      // Rejoin back into a string key
      let newKey = newTokens.joined(separator: " ")
      newWordFreqs[newKey, default: 0] += freq
    }
    
    return newWordFreqs
  }
}
