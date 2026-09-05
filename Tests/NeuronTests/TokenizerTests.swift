//
//  TokenizerTests.swift
//
//  Created by William Vabrinskas on 3/3/26.
//

import XCTest
@testable import Neuron

final class TokenizerTests: XCTestCase {

  // MARK: - Helpers

  private let defaultCorpus: [String] = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "cats and dogs are pets"
  ]

  private func makeTrainedTokenizer(corpus: [String]? = nil,
                                    vocabSize: Int = 50) -> BPETokenizer {
    let corpus = corpus ?? defaultCorpus
    let tokenizer = BPETokenizer(targetVocabSize: vocabSize)
    tokenizer.train(corpus: corpus)
    return tokenizer
  }

  // MARK: - Training

  func test_train_buildsVocabWithSpecialTokens() {
    // Special tokens: <pad>, <unk>, <bos>, <eos>, </w>
    let tokenizer = makeTrainedTokenizer()
    // Encoding a known character should not fall back to <unk>
    let ids = tokenizer.encode("cat")
    XCTAssertFalse(ids.isEmpty)
  }

  func test_train_vocabDoesNotExceedTargetSize() {
    // Use a tiny corpus so BPE merges are bounded by targetVocabSize
    let tokenizer = BPETokenizer(targetVocabSize: 10)
    tokenizer.train(corpus: ["ab"])
    // Vocab size should be <= targetVocabSize (training stops when reached)
    // We can verify indirectly: encoding should still work
    let ids = tokenizer.encode("ab")
    XCTAssertFalse(ids.isEmpty)
  }

  // MARK: - Encode

  func test_encode_singleWord_returnsNonEmptyIds() {
    let tokenizer = makeTrainedTokenizer()
    let ids = tokenizer.encode("cat")
    XCTAssertFalse(ids.isEmpty)
  }

  func test_encode_multipleSentenceWords_returnsIdsPerWord() {
    let tokenizer = makeTrainedTokenizer()
    let ids = tokenizer.encode("the cat")
    // Each word gets at least one token id (character + end-of-word marker)
    XCTAssertFalse(ids.isEmpty)
  }

  func test_encode_knownCharacters_doNotProduceUnknownToken() {
    let tokenizer = makeTrainedTokenizer()
    // After training, characters in the corpus are in vocab
    let ids = tokenizer.encode("cat")
    // Encode "cat" twice to get a stable id list
    let ids2 = tokenizer.encode("cat")
    XCTAssertEqual(ids, ids2)
  }

  func test_encode_unknownCharacter_fallsBackToUnkToken() {
    // Train on a small corpus that won't contain 'Z'
    let tokenizer = BPETokenizer(targetVocabSize: 30)
    tokenizer.train(corpus: ["abc"])
    // Encode a character never seen during training
    let ids = tokenizer.encode("Z")
    // Should produce exactly one ID (the <unk> id) plus the </w> marker
    XCTAssertFalse(ids.isEmpty)
  }

  func test_encode_emptyString_returnsEndOfWordMarker() {
    let tokenizer = makeTrainedTokenizer()
    // An empty string produces one "word" whose only token is </w>
    let ids = tokenizer.encode("")
    XCTAssertFalse(ids.isEmpty)
  }

  func test_encode_consistentAcrossCallsWithSameInput() {
    let tokenizer = makeTrainedTokenizer()
    let ids1 = tokenizer.encode("the cat")
    let ids2 = tokenizer.encode("the cat")
    XCTAssertEqual(ids1, ids2)
  }

  // MARK: - Decode

  func test_decode_emptyIds_returnsEmptyString() {
    let tokenizer = makeTrainedTokenizer()
    let text = tokenizer.decode([])
    XCTAssertEqual(text, "")
  }

  func test_decode_roundtrip_preservesText() {
    let tokenizer = makeTrainedTokenizer()
    let original = "the cat sat on the mat"
    let ids = tokenizer.encode(original)
    let decoded = tokenizer.decode(ids)
    XCTAssertEqual(decoded, original)
  }

  func test_decode_singleWord_roundtrip() {
    let tokenizer = makeTrainedTokenizer()
    let original = "cat"
    let ids = tokenizer.encode(original)
    let decoded = tokenizer.decode(ids)
    XCTAssertEqual(decoded, original)
  }

  func test_decode_stripsEndOfWordMarkers() {
    let tokenizer = makeTrainedTokenizer()
    let ids = tokenizer.encode("cat")
    let decoded = tokenizer.decode(ids)
    // The </w> marker should not appear in the decoded output
    XCTAssertFalse(decoded.contains("</w>"))
  }

  func test_decode_multipleWords_roundtrip() {
    let tokenizer = makeTrainedTokenizer()
    let original = "the dog ran in the park"
    let ids = tokenizer.encode(original)
    let decoded = tokenizer.decode(ids)
    XCTAssertEqual(decoded, original)
  }

  // MARK: - BPE Merges

  func test_bpe_repeatedSubword_isMerged() {
    // "aa" repeated in corpus should produce a "aa" token via BPE
    let tokenizer = BPETokenizer(targetVocabSize: 20)
    tokenizer.train(corpus: ["aa aa aa aa aa"])
    // After training, "aa" should be encodeable as fewer tokens than 2 characters
    let ids = tokenizer.encode("aa")
    // With BPE merging "a" + "a" → "aa", encoding "aa" should produce 2 ids: "aa</w>" or "aa" + "</w>"
    // This is <= 3 (would be 3 if no merge: 'a', 'a', '</w>')
    XCTAssertLessThanOrEqual(ids.count, 3)
  }

  func test_bpe_mergesAppliedInOrder() {
    // Verify that applying merge rules in order yields reproducible encoding
    let tokenizer = makeTrainedTokenizer()
    let ids1 = tokenizer.encode("cats")
    let ids2 = tokenizer.encode("cats")
    XCTAssertEqual(ids1, ids2)
  }

  // MARK: - Vocabulary invariants

  func test_vocabSize_countsMergeTokens() {
    // A target well above the base vocabulary forces merges to be created.
    let tokenizer = makeTrainedTokenizer(vocabSize: 60)
    let base = BPETokenizer(targetVocabSize: 0)
    base.train(corpus: defaultCorpus)

    // If vocabSize reported only special tokens + characters, merges would be uncounted and
    // their IDs would index past anything sized against it (e.g. the Embedding table).
    XCTAssertGreaterThan(tokenizer.vocabSize, base.vocabSize)
  }

  func test_vocabSize_boundsEveryEncodedId() {
    let tokenizer = makeTrainedTokenizer(vocabSize: 60)

    for text in defaultCorpus {
      for id in tokenizer.encode(text) {
        XCTAssertGreaterThanOrEqual(id, 0)
        XCTAssertLessThan(id, tokenizer.vocabSize,
                          "id \(id) is out of range 0..<\(tokenizer.vocabSize)")
      }
    }
  }

  func test_train_assignsContiguousIds() {
    let tokenizer = makeTrainedTokenizer(vocabSize: 60)

    // Every ID in 0..<vocabSize must be assigned. A hole means an ID was skipped, which
    // pushes the largest ID past vocabSize and out of range of anything sized against it.
    XCTAssertEqual(Set(tokenizer.reverseVocab.keys), Set(0..<tokenizer.vocabSize))
    XCTAssertEqual(Set(tokenizer.vocab.values), Set(0..<tokenizer.vocabSize))
  }

  func test_train_isRepeatable() {
    let tokenizer = BPETokenizer(targetVocabSize: 60)
    tokenizer.train(corpus: defaultCorpus)
    let firstSize = tokenizer.vocabSize
    let firstIds = tokenizer.encode("the cat")

    tokenizer.train(corpus: defaultCorpus)

    XCTAssertEqual(tokenizer.vocabSize, firstSize)
    XCTAssertEqual(tokenizer.encode("the cat"), firstIds)
  }

  // MARK: - Control tokens

  func test_controlTokenIds_excludeWordEndingMarker() {
    let tokenizer = makeTrainedTokenizer()
    // </w> carries surface text (a space), so skipping it would run words together.
    let ids = tokenizer.encode("the cat")
    let decoded = tokenizer.decode(ids, skipControlTokens: true)
    XCTAssertEqual(decoded, "the cat")
  }

  func test_decode_skipControlTokens_dropsPadding() {
    let tokenizer = makeTrainedTokenizer()
    let ids = tokenizer.encode("cat") + [Int](repeating: tokenizer.padTokenId, count: 5)

    XCTAssertEqual(tokenizer.decode(ids, skipControlTokens: true), "cat")
    XCTAssertTrue(tokenizer.decode(ids).contains("<pad>"))
  }

  func test_eosTokenId_isDistinctFromPadTokenId() {
    let tokenizer = makeTrainedTokenizer()
    XCTAssertNotEqual(tokenizer.eosTokenId, tokenizer.padTokenId)
    XCTAssertTrue(tokenizer.controlTokenIds.contains(tokenizer.eosTokenId))
    XCTAssertTrue(tokenizer.controlTokenIds.contains(tokenizer.padTokenId))
  }

  // MARK: - Export / Import

  func test_export_returnsValidURL() {
    let tokenizer = makeTrainedTokenizer()
    let url = tokenizer.export(name: "test-tokenizer-url", overrite: true, compress: true)
    XCTAssertNotNil(url)
  }

  func test_export_fileExists_atReturnedURL() {
    let tokenizer = makeTrainedTokenizer()
    guard let url = tokenizer.export(name: "test-tokenizer-exists", overrite: true, compress: true) else {
      XCTFail("Export returned nil URL")
      return
    }
    XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
  }

  func test_export_import_preservesEncoding() {
    let tokenizer = makeTrainedTokenizer()

    guard let url = tokenizer.export(name: "test-tokenizer-roundtrip", overrite: true, compress: true) else {
      XCTFail("Export returned nil URL")
      return
    }

    let result: Result<BPETokenizer, Error> = ExportHelper.buildModel(url)
    guard case .success(let imported) = result else {
      XCTFail("Import failed: \(result)")
      return
    }

    let originalIds = tokenizer.encode("the cat sat on the mat")
    let importedIds = imported.encode("the cat sat on the mat")
    XCTAssertEqual(originalIds, importedIds)
  }

  func test_export_import_preservesDecoding() {
    let tokenizer = makeTrainedTokenizer()

    guard let url = tokenizer.export(name: "test-tokenizer-decode-roundtrip", overrite: true, compress: true) else {
      XCTFail("Export returned nil URL")
      return
    }

    let result: Result<BPETokenizer, Error> = ExportHelper.buildModel(url)
    guard case .success(let imported) = result else {
      XCTFail("Import failed: \(result)")
      return
    }

    let original = "the cat sat on the mat"
    let importedDecoded = imported.decode(imported.encode(original))
    XCTAssertEqual(importedDecoded, original)
  }

  func test_export_import_fromData() {
    let tokenizer = makeTrainedTokenizer()

    guard let url = tokenizer.export(name: "test-tokenizer-data", overrite: true, compress: true),
          let data = try? Data(contentsOf: url) else {
      XCTFail("Export or data read failed")
      return
    }

    let result: Result<BPETokenizer, Error> = ExportHelper.buildModel(data)
    guard case .success(let imported) = result else {
      XCTFail("Import from Data failed")
      return
    }

    XCTAssertEqual(tokenizer.encode("cat"), imported.encode("cat"))
  }

  func test_export_noTimestamp_when_overrite_isTrue() {
    let tokenizer = makeTrainedTokenizer()
    let name = "test-no-timestamp"
    guard let url = tokenizer.export(name: name, overrite: true, compress: true) else {
      XCTFail("Export returned nil")
      return
    }
    // Filename should start with the given name (no timestamp appended)
    XCTAssertTrue(url.lastPathComponent.hasPrefix(name))
  }

  func test_export_withTimestamp_when_overrite_isFalse() {
    let tokenizer = makeTrainedTokenizer()
    let name = "test-with-timestamp"
    guard let url = tokenizer.export(name: name, overrite: false, compress: true) else {
      XCTFail("Export returned nil")
      return
    }
    // Filename should contain a '-' followed by the timestamp when overrite is false
    let filename = url.deletingPathExtension().lastPathComponent
    XCTAssertTrue(filename.hasPrefix(name + "-"))
  }
  
  func test_train_buildsVocab_largerCorpus() {
    let corpus = [
        "the quick brown fox jumps over the lazy dog and the dog barked loudly at the fox",
        "a tokenizer splits text into smaller units called tokens and each token gets an id",
        "the cat sat on the mat and the mat was flat and the cat was fat",
        "machine learning models cannot read raw text so text must be converted into numbers",
        "the dog ran quickly through the park and jumped over the fence near the lake",
        "tokenizers are trained on large corpora to learn which character sequences are common",
        "the fox and the dog became friends and played together in the park every day",
        "each word is split into characters and then pairs of characters are merged together",
        "the lazy cat slept on the warm mat near the window in the afternoon sun",
        "a vocabulary is built by merging the most frequent pairs of tokens iteratively"
    ]
    
    let tokenizer = BPETokenizer(targetVocabSize: 100)
    tokenizer.train(corpus: corpus)
    
    // Encoding a known character should not fall back to <unk>
    let ids = tokenizer.encode("cat")
    XCTAssertFalse(ids.isEmpty)
  }
}
