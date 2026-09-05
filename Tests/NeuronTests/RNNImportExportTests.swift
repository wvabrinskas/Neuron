//
//  RNNImportExportTests.swift
//
//  Verifies that exporting an RNN and importing it restores the exact state: the same
//  vocabulary, the same sequence length, the same weights, and the same predictions.
//

import XCTest
import NumSwift
@testable import Neuron

/// Minimal dataset whose `build()` re-trains its tokenizer, which is what makes the import
/// path interesting: a naive import would let that re-training replace the imported vocabulary.
private final class ImportTestDataset: TokenizableDataset {
  var corpus: [String] = ["hammley", "spammley", "Dugley", "Absoluteley"]
  var sampleCount = 8

  override func build() async -> TokenizingDatasetData {
    tokenizer.train(corpus: corpus)

    let length = sequenceLength(for: corpus)
    let pairs = corpus.map { nextTokenPair(for: $0, sequenceLength: length) }

    var training: [DatasetModel] = []
    var val: [DatasetModel] = []
    for pair in pairs {
      let model = DatasetModel(data: pair.data, label: pair.label)
      training.append(contentsOf: [DatasetModel](repeating: model, count: sampleCount))
      val.append(model)
    }
    return (training, val)
  }
}

final class RNNImportExportTests: XCTestCase {

  private func makeRNN(dataset: ImportTestDataset) -> RNN<ImportTestDataset> {
    RNN(returnSequence: true,
        dataset: dataset,
        classifierParameters: .init(batchSize: 8, epochs: 1, killOnAccuracy: false),
        optimizerParameters: .init(learningRate: 0.001),
        lstmParameters: .init(hiddenUnits: 16, inputUnits: 8))
  }

  func test_importFrom_restoresVocabularyAndSequenceLength() async {
    let original = makeRNN(dataset: ImportTestDataset(tokenizer: .init(targetVocabSize: 24)))
    await original.readyUp()

    let exported = original.exportWithVectors(overrite: true, compress: true)
    guard let modelURL = exported.model, let vectorsURL = exported.vectors else {
      XCTFail("export produced no files")
      return
    }

    // A fresh trainer whose tokenizer is untrained until the import supplies one.
    let imported = makeRNN(dataset: ImportTestDataset(tokenizer: .init(targetVocabSize: 24)))
    await imported.importFrom(url: modelURL, vectors: vectorsURL)

    XCTAssertGreaterThan(imported.vocabSize, 0)
    XCTAssertEqual(imported.vocabSize, original.vocabSize)
    XCTAssertEqual(imported.wordLength, original.wordLength)
    XCTAssertEqual(imported.optimizer.ignoreLabelIndex, original.optimizer.ignoreLabelIndex)
  }

  func test_importFrom_restoresTheImportedNetworkNotAFreshOne() async {
    let original = makeRNN(dataset: ImportTestDataset(tokenizer: .init(targetVocabSize: 24)))
    await original.readyUp()

    let exported = original.exportWithVectors(overrite: true, compress: true)
    guard let modelURL = exported.model, let vectorsURL = exported.vectors else {
      XCTFail("export produced no files")
      return
    }

    let imported = makeRNN(dataset: ImportTestDataset(tokenizer: .init(targetVocabSize: 24)))
    await imported.importFrom(url: modelURL, vectors: vectorsURL)

    // The layers the trainer reports must be the ones actually in the optimizer's graph.
    // The previous import compiled a throwaway network, replaced it, and kept describing the
    // discarded layers.
    let graphLayers = imported.optimizer.trainable.layers
    XCTAssertTrue(graphLayers.contains { $0 === imported.embedding })
    XCTAssertTrue(graphLayers.contains { $0 === imported.lstm })
  }

  func test_importFrom_preservesWeights() async {
    let original = makeRNN(dataset: ImportTestDataset(tokenizer: .init(targetVocabSize: 24)))
    await original.readyUp()

    let exported = original.exportWithVectors(overrite: true, compress: true)
    guard let modelURL = exported.model, let vectorsURL = exported.vectors else {
      XCTFail("export produced no files")
      return
    }

    let imported = makeRNN(dataset: ImportTestDataset(tokenizer: .init(targetVocabSize: 24)))
    await imported.importFrom(url: modelURL, vectors: vectorsURL)

    guard let originalEmbedding = original.embedding,
          let importedEmbedding = imported.embedding else {
      XCTFail("missing embedding layer")
      return
    }

    XCTAssertEqual(importedEmbedding.weights.shape, originalEmbedding.weights.shape)
    XCTAssertEqual(importedEmbedding.batchLength, originalEmbedding.batchLength)

    let a = originalEmbedding.weights.storage
    let b = importedEmbedding.weights.storage
    XCTAssertEqual(a.count, b.count)
    for i in 0..<Swift.min(a.count, b.count) {
      XCTAssertEqual(a[i], b[i], accuracy: 1e-5, "embedding weight \(i) differs")
    }
  }

  func test_importFrom_producesIdenticalPredictions() async {
    let original = makeRNN(dataset: ImportTestDataset(tokenizer: .init(targetVocabSize: 24)))
    await original.readyUp()

    let exported = original.exportWithVectors(overrite: true, compress: true)
    guard let modelURL = exported.model, let vectorsURL = exported.vectors else {
      XCTFail("export produced no files")
      return
    }

    let imported = makeRNN(dataset: ImportTestDataset(tokenizer: .init(targetVocabSize: 24)))
    await imported.importFrom(url: modelURL, vectors: vectorsURL)

    // Same input through both graphs must give the same distribution.
    let input = original.dataset.nextTokenPair(for: "hammley",
                                               sequenceLength: original.wordLength).data

    let a = original.optimizer.predict([input])[0].storage
    let b = imported.optimizer.predict([input])[0].storage

    XCTAssertEqual(a.count, b.count)
    XCTAssertGreaterThan(a.count, 0)
    for i in 0..<Swift.min(a.count, b.count) {
      XCTAssertEqual(a[i], b[i], accuracy: 1e-5, "output \(i) differs")
    }
  }
}
