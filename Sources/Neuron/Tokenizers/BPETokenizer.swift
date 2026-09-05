//
//  Tokenizing.swift
//  Neuron
//
//  Created by William Vabrinskas on 2/18/26.
//

import NumSwift
import Foundation


/// A base Byte Pair Encoding (BPE) tokenizer that learns subword merge rules from a text corpus.
///
/// Subclasses may override `train(_:)`, `encode(_:)`, and `decode(_:)` to customise tokenization behaviour.
/// The learned vocabulary and merge rules are serializable via the `Exportable` protocol.
open class BPETokenizer: Tokenizing {
  /// The number of tokens in the trained vocabulary.
  ///
  /// This counts merge tokens as well as the base characters and control tokens, so it is
  /// always one past the largest assigned ID. Anything that sizes a layer against the
  /// vocabulary (`Embedding`, `LSTM`) depends on that: reporting only the base vocabulary
  /// would let merge IDs index past the end of the embedding table.
  ///
  /// Returns `0` until `train(corpus:)` has run.
  public var vocabSize: Int {
    vocab.count
  }

  /// The token ID used to pad sequences out to a fixed length.
  public var padTokenId: Int {
    vocab[wordPad] ?? 0
  }

  /// The token ID marking the end of a generated sequence.
  public var eosTokenId: Int {
    vocab[wordEos] ?? 0
  }

  /// IDs of tokens that carry no surface text -- padding, unknown, and the sequence markers.
  ///
  /// `wordEnding` is deliberately excluded: it decodes to a space and is part of the text.
  public var controlTokenIds: Set<Int> {
    Set(controlTokens.compactMap { vocab[$0] })
  }

  /// The next unassigned token ID.
  ///
  /// Derived from the vocabulary rather than stored, which keeps IDs contiguous in
  /// `0..<vocabSize` across training, decoding, and retraining. A stored counter drifts:
  /// seeding it from `Vectorizer.lastKey + 1` skipped an ID, because `lastKey` is already
  /// the next free slot.
  private var nextId: Int {
    vocab.count
  }
  
  /// Token -> ID. Internal rather than private so tests can assert on the ID space directly;
  /// a decode round-trip can't distinguish an unassigned ID from one that renders as whitespace.
  private(set) var vocab: [String: Int] = [:]
  private(set) var reverseVocab: [Int: String] = [:]
  
  private let vectorizer = Vectorizer()
  
  private var corpus: TokenizerCorpus = []
  
  private let wordEnding: String = "</w>"
  private let wordUnknown: String = "<unk>"
  private let wordPad: String = "<pad>"
  private let wordBos: String = "<bos>"
  private let wordEos: String = "<eos>"
  private let targetVocabSize: Int
  private var mergeRules: [TokenPair] = []

  private struct TokenPair: Hashable, Codable {
    var tokenA: String
    var tokenB: String
    
    func join() -> String {
      tokenA + tokenB
    }
  }
  
  /// Tokens with no surface text. Skipped when decoding with `skipControlTokens`.
  private lazy var controlTokens: [String] = [
      wordPad,
      wordUnknown,
      wordBos,
      wordEos
  ]

  private lazy var specialTokens: [String] = controlTokens + [wordEnding]
  
  /// Coding keys used for encoding and decoding the tokenizer's properties.
  public enum CodingKeys: String, CodingKey {
    /// Key for the array of learned byte-pair merge rules.
    case mergeRules
    /// Key for the token-to-ID vocabulary dictionary.
    case vocab
    /// Key for the ID-to-token inverse vocabulary dictionary.
    case inverseVocab
    /// Key for the target vocabulary size used during training.
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
  
  /// Trains the BPE tokenizer on the given text corpus, building a vocabulary up to `targetVocabSize`.
  ///
  /// - Parameter corpus: An array of text strings used to learn byte-pair merge rules.
  open func train(corpus: TokenizerCorpus) {
    // Training is not incremental. Resetting the merge rules -- and reseeding `vocab` from the
    // vectorizer below, which drops any merge tokens from a previous run -- keeps a second
    // call from stacking stale rules on top of a rebuilt vocabulary.
    mergeRules.removeAll()

    let flatCorpus = corpus.joined(separator: " ")
    
    vectorizer.vectorize(specialTokens)
    vectorizer.vectorize(flatCorpus.characters)
    vocab = vectorizer.vector
    reverseVocab = vectorizer.inverseVector
    
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
      let newId = nextId
      vocab[newToken] = newId
      reverseVocab[newId] = newToken
      
      mergeRules.append(bestPair)
      
      wordFrequency = applyMerge(wordFreqs: wordFrequency,
                                 pair: bestPair)
    }
  }
  
  /// Encodes a text string into a sequence of token IDs using the learned BPE vocabulary.
  ///
  /// - Parameter text: The input string to encode.
  /// - Returns: An array of integer token IDs corresponding to the input text.
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
  
  /// Decodes a sequence of token IDs back into a human-readable string.
  ///
  /// - Parameter tokenIds: An array of integer token IDs to decode.
  /// - Returns: The reconstructed string with end-of-word markers replaced by spaces.
  open func decode(_ tokenIds: [Int]) -> String {
    decode(tokenIds, skipControlTokens: false)
  }

  /// Decodes a sequence of token IDs back into a human-readable string.
  ///
  /// - Parameters:
  ///   - tokenIds: An array of integer token IDs to decode.
  ///   - skipControlTokens: When `true`, padding and sequence markers are dropped rather than
  ///     rendered as their literal names. Generated text wants this; a round-trip check does not.
  /// - Returns: The reconstructed string with end-of-word markers replaced by spaces.
  open func decode(_ tokenIds: [Int], skipControlTokens: Bool) -> String {
    // Invert the vocab dictionary
    let idToToken = reverseVocab
    let skipped = skipControlTokens ? controlTokenIds : []

    // Map IDs back to token strings
    let tokens = tokenIds
      .filter { skipped.contains($0) == false }
      .compactMap { idToToken[$0] }
    
    // Join and clean up end-of-word markers
    return tokens
      .joined()
      .replacingOccurrences(of: wordEnding, with: " ")
      .trimmingCharacters(in: .whitespaces)
  }
  
  
  /// Exports the trainable as a `.stkns` file.
  ///
  /// - Parameters:
  ///   - name: Optional filename prefix.
  ///   - overrite: When `false`, appends a timestamp to avoid overwrite.
  ///   - compress: When `true`, emits compact JSON.
  /// - Returns: URL to the exported model file, or `nil` on write failure.
  @discardableResult
  public func export(name: String?, overrite: Bool, compress: Bool) -> URL? {
    let additional = overrite == false ? "-\(Date().timeIntervalSince1970)" : ""
    
    let filename = (name ?? "tokens") + additional
    
    let dUrl = ExportHelper.getTokens(filename: filename, compress: compress, model: self)
    
    return dUrl
  }
  
  /// Reconstructs a `Sequential` model directly from encoded model data.
  ///
  /// - Parameter data: Serialized model bytes.
  /// - Returns: Decoded `Sequential` instance.
  public static func `import`(_ data: Data) -> Self {
    let result: Result<Self, Error> =  ExportHelper.buildModel(data)
    switch result {
    case .success(let model):
      return model
    case .failure(let error):
      preconditionFailure(error.localizedDescription)
    }
  }
  
  /// Reconstructs a `Vectorizer` model from a serialized `.stkns` file URL.
  ///
  /// - Parameter url: File URL pointing to a previously exported model.
  /// - Returns: Decoded `Sequential` instance.
  public static func `import`(_ url: URL) -> Self {
    let result: Result<Self, Error> =  ExportHelper.buildModel(url)
    switch result {
    case .success(let model):
      return model
    case .failure(let error):
      preconditionFailure(error.localizedDescription)
    }
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
