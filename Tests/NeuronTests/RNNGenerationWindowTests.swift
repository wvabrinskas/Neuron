//
//  RNNGenerationWindowTests.swift
//
//  `LSTM.forward` reads a fixed window `0..<batchLength` from the start of its input and never
//  looks past it. Generation therefore has to slide that window over the produced tokens; feeding
//  the whole history freezes the context and repeats one token until `maxTokenCount`.
//

import XCTest
import NumSwift
@testable import Neuron

/// Records every tensor handed to the network and returns a distribution that depends on the
/// window's *last* token, so a frozen context is directly observable in the output.
private final class SpyTrainable: Trainable {
  var name: String = "spy"
  var layers: [Layer] = []
  var isCompiled: Bool = true
  var isTraining: Bool = true
  var deviceType: DeviceType = .cpu
  var device: Device = CPU()
  var batchSize: Int = 1

  let vocabSize: Int
  let batchLength: Int
  /// Depth of every input tensor the network was asked to run.
  var seenDepths: [Int] = []
  /// Token IDs of every input tensor the network was asked to run.
  var seenWindows: [[Int]] = []

  init(vocabSize: Int, batchLength: Int) {
    self.vocabSize = vocabSize
    self.batchLength = batchLength
  }

  func predict(_ data: Tensor, context: NetworkContext) -> Tensor {
    let ids = data.storage.map { Int($0) }
    seenDepths.append(data.size.depth)
    seenWindows.append(ids)

    // Emit `batchLength` timesteps, as the real LSTM does with returnSequence. Each timestep
    // puts all its mass on (lastToken + 1) % vocabSize, so a sliding window produces a walking
    // sequence and a frozen window produces the same token forever.
    let last = ids.last ?? 0
    let hot = (last + 1) % vocabSize

    let storage = TensorStorage.create(count: vocabSize * batchLength)
    for d in 0..<batchLength {
      storage[d * vocabSize + hot] = 1
    }
    return Tensor(storage: storage,
                  size: TensorSize(rows: 1, columns: vocabSize, depth: batchLength))
  }

  func predict(batch: TensorBatch, context: NetworkContext) -> TensorBatch {
    batch.map { predict($0, context: context) }
  }

  func compile() {}
  func exportWeights() throws -> [[Tensor]] { [] }
  func importWeights(_ weights: [[Tensor]]) throws {}
  func apply(gradients: Tensor.Gradient, learningRate: Tensor.Scalar) {}
  func export(name: String?, overrite: Bool, compress: Bool) -> URL? { nil }
  static func `import`(_ url: URL) -> Self { fatalError("unused") }
  var debugDescription: String { "SpyTrainable" }

  init(from decoder: Decoder) throws { fatalError("unused") }
  func encode(to encoder: Encoder) throws {}
}

private final class WindowTestDataset: TokenizableDataset {
  let corpus = ["hammley", "spammley", "Dugley"]

  override func build() async -> TokenizingDatasetData {
    tokenizer.train(corpus: corpus)
    let length = sequenceLength(for: corpus)
    let pairs = corpus.map { nextTokenPair(for: $0, sequenceLength: length) }
    let models = pairs.map { DatasetModel(data: $0.data, label: $0.label) }
    return (models, models)
  }
}

final class RNNGenerationWindowTests: XCTestCase {

  private func makeReadyRNN() async -> RNN<WindowTestDataset> {
    let rnn = RNN(returnSequence: true,
                  dataset: WindowTestDataset(tokenizer: .init(targetVocabSize: 24)),
                  classifierParameters: .init(batchSize: 3, epochs: 1, killOnAccuracy: false),
                  optimizerParameters: .init(learningRate: 0.001),
                  lstmParameters: .init(hiddenUnits: 8, inputUnits: 4))
    await rnn.readyUp()
    return rnn
  }

  func test_predict_neverFeedsMoreTimestepsThanTheCompiledWindow() async {
    let rnn = await makeReadyRNN()
    let spy = SpyTrainable(vocabSize: rnn.vocabSize, batchLength: rnn.wordLength)
    rnn.optimizer.trainable = spy

    // Deliberately far past the compiled sequence length.
    _ = rnn.predict(count: 1, maxTokenCount: rnn.wordLength * 3, randomizeSelection: false)

    XCTAssertFalse(spy.seenDepths.isEmpty, "the network was never asked to predict")
    for depth in spy.seenDepths {
      XCTAssertLessThanOrEqual(depth, rnn.wordLength,
                               "LSTM ignores depth past batchLength; feeding \(depth) wastes it")
    }
  }

  func test_predict_slidesTheWindowInsteadOfFreezingIt() async {
    let rnn = await makeReadyRNN()
    let spy = SpyTrainable(vocabSize: rnn.vocabSize, batchLength: rnn.wordLength)
    rnn.optimizer.trainable = spy

    _ = rnn.predict(count: 1, maxTokenCount: rnn.wordLength * 3, randomizeSelection: false)

    // Once the window is full, each successive call must drop the oldest token and gain the
    // newest. A frozen context would repeat the identical window every iteration.
    let full = spy.seenWindows.filter { $0.count == rnn.wordLength }
    XCTAssertGreaterThan(full.count, 1, "window never filled; test is not exercising the case")

    for (previous, next) in zip(full, full.dropFirst()) {
      XCTAssertNotEqual(previous, next, "context froze: the same window was fed twice")
      XCTAssertEqual(Array(previous.dropFirst()), Array(next.dropLast()),
                     "window should advance by exactly one token")
    }
  }

  func test_predict_generationDoesNotStallOnOneToken() async {
    let rnn = await makeReadyRNN()
    let spy = SpyTrainable(vocabSize: rnn.vocabSize, batchLength: rnn.wordLength)
    rnn.optimizer.trainable = spy

    _ = rnn.predict(count: 1, maxTokenCount: rnn.wordLength * 3, randomizeSelection: false)

    // The spy walks the vocabulary, so a working sliding window yields distinct tokens. With a
    // frozen context every generated token past the first window is identical.
    let generated = spy.seenWindows.last ?? []
    XCTAssertGreaterThan(Set(generated).count, 1,
                         "generation collapsed to a single repeated token: \(generated)")
  }
}
