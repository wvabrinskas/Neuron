//
//  TokenizableDatasetTests.swift
//
//  Covers next-token pair construction and the tensor layout the Embedding depends on.
//

import XCTest
@testable import Neuron

final class TokenizableDatasetTests: XCTestCase {

  private let corpus = ["the cat sat on the mat",
                        "the dog ran in the park",
                        "cats and dogs are pets",
                        "a small dog naps"]

  private func makeDataset(vocab: Int = 60) -> (TokenizableDataset, BPETokenizer) {
    let tokenizer = BPETokenizer(targetVocabSize: vocab)
    tokenizer.train(corpus: corpus)
    return (TokenizableDataset(tokenizer: tokenizer), tokenizer)
  }

  // MARK: - Tensor layout

  func test_tokenize_putsOneTokenPerDepthSlice() {
    let (dataset, tokenizer) = makeDataset()
    let text = corpus[0]

    let tensor = dataset.tokenize(text)

    // Embedding reads one index from each depth slice. Packing IDs along `columns` would
    // leave the tensor at depth 1 and the model would only ever see the first token.
    XCTAssertEqual(tensor.size.depth, tokenizer.encode(text).count)
    XCTAssertEqual(tensor.size.rows, 1)
    XCTAssertEqual(tensor.size.columns, 1)
  }

  // MARK: - Next-token pairs

  func test_nextTokenPair_labelIsInputShiftedByOneToken() {
    let (dataset, _) = makeDataset()
    let length = dataset.sequenceLength(for: corpus)

    let pair = dataset.nextTokenPair(for: corpus[0], sequenceLength: length)
    let input = pair.data.storage.map { Int($0) }
    let label = pair.label.storage.map { Int($0) }

    XCTAssertEqual(input.count, length)
    XCTAssertEqual(label.count, length)

    // label[i] must be the token that follows input[i].
    for i in 0..<(length - 1) {
      XCTAssertEqual(label[i], input[i + 1], "label[\(i)] should equal input[\(i + 1)]")
    }
  }

  func test_nextTokenPair_textShiftWouldNotProduceThis() {
    let (dataset, tokenizer) = makeDataset()
    let length = dataset.sequenceLength(for: corpus)

    let pair = dataset.nextTokenPair(for: corpus[0], sequenceLength: length)
    let label = pair.label.storage.map { Int($0) }

    // Dropping the first *character* and re-encoding leaves every position after the first
    // identical to the input, which trains the model to copy. The token shift must differ.
    let textShifted = tokenizer.encode(String(corpus[0].dropFirst()))
    XCTAssertNotEqual(Array(label.prefix(textShifted.count)), textShifted)
  }

  func test_nextTokenPair_wrapsInBoundaryTokens() {
    let (dataset, tokenizer) = makeDataset()
    let length = dataset.sequenceLength(for: corpus)

    let pair = dataset.nextTokenPair(for: corpus[0], sequenceLength: length)
    let input = pair.data.storage.map { Int($0) }
    let label = pair.label.storage.map { Int($0) }

    XCTAssertEqual(input.first, tokenizer.bosTokenId)
    XCTAssertTrue(label.contains(tokenizer.eosTokenId))
  }

  func test_nextTokenPair_padsShortItemsInBothTensors() {
    let (dataset, tokenizer) = makeDataset()
    let length = dataset.sequenceLength(for: corpus)

    // The shortest item needs padding to reach the shared length.
    let shortest = corpus.min(by: { dataset.tokenCount(for: $0) < dataset.tokenCount(for: $1) })!
    let pair = dataset.nextTokenPair(for: shortest, sequenceLength: length)

    XCTAssertEqual(pair.data.size.depth, length)
    XCTAssertEqual(pair.label.size.depth, length)
    XCTAssertEqual(pair.data.storage.last.map { Int($0) }, tokenizer.padTokenId)
    XCTAssertEqual(pair.label.storage.last.map { Int($0) }, tokenizer.padTokenId)
  }

  func test_nextTokenPair_truncatesWithoutLosingTheFinalLabel() {
    let (dataset, _) = makeDataset()

    let pair = dataset.nextTokenPair(for: corpus[0], sequenceLength: 3)
    let input = pair.data.storage.map { Int($0) }
    let label = pair.label.storage.map { Int($0) }

    XCTAssertEqual(input.count, 3)
    XCTAssertEqual(label.count, 3)
    // Truncation happens at sequenceLength + 1 IDs, so the shift still holds at every position.
    for i in 0..<2 {
      XCTAssertEqual(label[i], input[i + 1])
    }
  }

  func test_sequenceLength_fitsLongestItemWithoutTruncation() {
    let (dataset, _) = makeDataset()
    let length = dataset.sequenceLength(for: corpus)

    // A run of n tokens yields n - 1 pairs: the last token is only ever a label.
    let longest = corpus.map { dataset.tokenCount(for: $0) }.max()!
    XCTAssertEqual(length, longest - 1)
  }

  func test_sequenceLength_respectsCap() {
    let (dataset, _) = makeDataset()
    XCTAssertEqual(dataset.sequenceLength(for: corpus, cappedAt: 2), 2)
  }

  // MARK: - Decoding

  func test_item_decodesWholeSequenceWithWordBoundaries() {
    let (dataset, tokenizer) = makeDataset()
    let ids = tokenizer.encode(corpus[0])

    let whole = dataset.item(for: Tensor(ids.map { [[Tensor.Scalar($0)]] }))
    let perToken = ids.map { dataset.item(for: Tensor(Tensor.Scalar($0))) }.joined()

    // `decode` turns </w> into a space and then trims, so per-token concatenation loses every
    // boundary. RNN.predict must decode the accumulated IDs in one pass.
    XCTAssertEqual(whole, corpus[0])
    XCTAssertFalse(perToken.contains(" "))
  }

  func test_item_skipsControlTokens() {
    let (dataset, tokenizer) = makeDataset()
    let ids = tokenizer.encode("the cat") + [tokenizer.padTokenId, tokenizer.eosTokenId]

    let decoded = dataset.item(for: Tensor(ids.map { [[Tensor.Scalar($0)]] }))

    XCTAssertEqual(decoded, "the cat")
  }
}
