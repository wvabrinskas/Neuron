//
//  LSTMGradientTests.swift
//  Neuron
//

@testable import Neuron
import XCTest

/// Covers the BPTT recurrent-error bound on `LSTM`, the Adam epsilon placement, and the
/// per-gate gradient norm inspector.
final class LSTMGradientTests: XCTestCase {
  
  private let inputUnits = 8
  private let hiddenUnits = 32
  private let vocabSize = 12
  private let batchLength = 24
  
  /// Builds an `Embedding -> LSTM` pair with gate weights scaled up so the recurrent
  /// Jacobian has gain > 1 and BPTT grows across the sequence.
  private func makeExplodingLSTM(recurrentErrorClip: Tensor.Scalar?) -> (embedding: Embedding, lstm: LSTM) {
    let embedding = Embedding(inputUnits: inputUnits,
                              vocabSize: vocabSize,
                              batchLength: batchLength)
    
    let lstm = LSTM(inputUnits: inputUnits,
                    batchLength: batchLength,
                    biasEnabled: true,
                    initializer: .xavierNormal,
                    hiddenUnits: hiddenUnits,
                    vocabSize: vocabSize,
                    recurrentErrorClip: recurrentErrorClip)
    
    func scaled(_ tensor: Tensor, by factor: Tensor.Scalar) -> Tensor {
      Tensor(storage: tensor.storage * factor, size: tensor.size)
    }
    
    lstm.forgetGateWeights = scaled(lstm.forgetGateWeights, by: 8)
    lstm.inputGateWeights = scaled(lstm.inputGateWeights, by: 8)
    lstm.gateGateWeights = scaled(lstm.gateGateWeights, by: 8)
    lstm.outputGateWeights = scaled(lstm.outputGateWeights, by: 8)
    
    return (embedding, lstm)
  }
  
  private func makeInput() -> Tensor {
    let indices = (0..<batchLength).map { [[Tensor.Scalar($0 % vocabSize)]] }
    return Tensor(indices)
  }
  
  /// Runs forward + backward and returns the LSTM's packed weight-gradient L2 norm.
  private func weightGradientNorm(embedding: Embedding, lstm: LSTM, delta: Tensor) -> Tensor.Scalar {
    let embedded = embedding.forward(tensor: makeInput())
    let out = lstm.forward(tensor: embedded, context: .init())
    out.setGraph(embedded)
    
    let gradients = out.gradients(delta: delta, wrt: embedded)
    
    // The LSTM is the only layer in the graph; its packed weight gradient is the non-empty entry.
    guard let packed = gradients.weights.first(where: { $0.isEmpty == false }) else {
      XCTFail("expected an LSTM weight gradient")
      return 0
    }
    return packed.l2Norm()
  }
  
  // MARK: - recurrentErrorClip
  
  func test_LSTM_recurrentErrorClip_boundsBackpropagatedGradient() {
    // Same weights for both runs: build once, then toggle the clip on the same instance.
    let (embedding, lstm) = makeExplodingLSTM(recurrentErrorClip: nil)
    let delta = Tensor.fillWith(value: 50, size: lstm.outputSize)
    
    let unclipped = weightGradientNorm(embedding: embedding, lstm: lstm, delta: delta)
    
    lstm.recurrentErrorClip = 1.0
    let clipped = weightGradientNorm(embedding: embedding, lstm: lstm, delta: delta)
    
    XCTAssertTrue(unclipped.isFinite)
    XCTAssertTrue(clipped.isFinite)
    XCTAssertGreaterThan(clipped, 0)
    XCTAssertLessThan(clipped, unclipped,
                      "bounding the carried hidden/cell error should reduce the packed weight gradient (clipped: \(clipped), unclipped: \(unclipped))")
  }
  
  func test_LSTM_recurrentErrorClip_nilLeavesGradientUnchanged() {
    let (embedding, lstm) = makeExplodingLSTM(recurrentErrorClip: nil)
    let delta = Tensor.fillWith(value: 50, size: lstm.outputSize)
    
    let first = weightGradientNorm(embedding: embedding, lstm: lstm, delta: delta)
    let second = weightGradientNorm(embedding: embedding, lstm: lstm, delta: delta)
    
    XCTAssertEqual(first, second, accuracy: first * 1e-4, "backward should be deterministic with no clip")
  }
  
  func test_LSTM_recurrentErrorClip_isSerialized() {
    func roundTrip(_ clip: Tensor.Scalar?) throws -> Tensor.Scalar? {
      let lstm = LSTM(inputUnits: inputUnits,
                      batchLength: batchLength,
                      hiddenUnits: hiddenUnits,
                      vocabSize: vocabSize,
                      recurrentErrorClip: clip)
      let data = try JSONEncoder().encode(lstm)
      return try JSONDecoder().decode(LSTM.self, from: data).recurrentErrorClip
    }
    
    XCTAssertEqual(try roundTrip(1.0), 1.0)
    XCTAssertEqual(try roundTrip(0.25), 0.25)
    XCTAssertNil(try roundTrip(nil))
  }
  
  func test_LSTM_recurrentErrorClip_defaultsWhenKeyMissing() throws {
    // Models exported before the key existed decode with the default (disabled).
    let lstm = LSTM(inputUnits: inputUnits,
                    batchLength: batchLength,
                    hiddenUnits: hiddenUnits,
                    vocabSize: vocabSize)
    let data = try JSONEncoder().encode(lstm)
    
    var json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    json.removeValue(forKey: "recurrentErrorClip")
    let legacy = try JSONSerialization.data(withJSONObject: json)
    
    let decoded = try JSONDecoder().decode(LSTM.self, from: legacy)
    XCTAssertNil(decoded.recurrentErrorClip)
  }
  
  // MARK: - Packed gradient layout
  
  /// The gate and hidden-output groups have different shapes. Packing them by depth produced a
  /// tensor whose declared size exceeded its storage; the accumulator's `+` then walked the
  /// declared size and read past the buffer, which showed up as an exploding global gradient
  /// norm coming from nowhere. Packed gradients must always be exactly their storage.
  func test_LSTM_packedGradients_declaredSizeMatchesStorage() {
    let (embedding, lstm) = makeExplodingLSTM(recurrentErrorClip: 1.0)
    let embedded = embedding.forward(tensor: makeInput())
    let out = lstm.forward(tensor: embedded, context: .init())
    out.setGraph(embedded)
    let gradients = out.gradients(delta: Tensor.fillRandom(size: out.size), wrt: embedded)
    
    let expectedWeights = 4 * (inputUnits + hiddenUnits) * hiddenUnits + vocabSize * hiddenUnits
    let expectedBiases = 4 * hiddenUnits + vocabSize
    
    let weights = try! XCTUnwrap(gradients.weights.first(where: { $0.isEmpty == false }))
    let biases = try! XCTUnwrap(gradients.biases.first(where: { $0.isEmpty == false }))
    
    XCTAssertEqual(weights.storage.count, expectedWeights)
    XCTAssertEqual(weights.shape.reduce(1, *), weights.storage.count)
    XCTAssertEqual(biases.storage.count, expectedBiases)
    XCTAssertEqual(biases.shape.reduce(1, *), biases.storage.count)
    
    // What the accumulator does to them must preserve the layout exactly.
    let summedBiases = biases + biases
    let averagedBiases = summedBiases / 2
    XCTAssertEqual(summedBiases.storage.count, expectedBiases)
    XCTAssertEqual(averagedBiases.storage.count, expectedBiases)
    XCTAssertEqual(averagedBiases.storage, biases.storage)
    XCTAssertEqual((weights + weights).storage.count, expectedWeights)
    
    // The layer's own packed views use the same layout.
    XCTAssertEqual(lstm.weights.storage.count, expectedWeights)
    XCTAssertEqual(lstm.weights.shape.reduce(1, *), expectedWeights)
    XCTAssertEqual(lstm.biases.storage.count, expectedBiases)
    XCTAssertEqual(lstm.biases.shape.reduce(1, *), expectedBiases)
  }
  
  func test_LSTM_apply_updatesEveryGroup() {
    let (embedding, lstm) = makeExplodingLSTM(recurrentErrorClip: 1.0)
    let embedded = embedding.forward(tensor: makeInput())
    let out = lstm.forward(tensor: embedded, context: .init())
    out.setGraph(embedded)
    let gradients = out.gradients(delta: Tensor.fillRandom(size: out.size), wrt: embedded)
    let weights = try! XCTUnwrap(gradients.weights.first(where: { $0.isEmpty == false }))
    let biases = try! XCTUnwrap(gradients.biases.first(where: { $0.isEmpty == false }))
    
    let before = [lstm.forgetGateWeights, lstm.inputGateWeights, lstm.gateGateWeights,
                  lstm.outputGateWeights, lstm.hiddenOutputWeights].map { $0.copy() }
    let beforeBiases = [lstm.forgetGateBiases, lstm.inputGateBiases, lstm.gateGateBiases,
                        lstm.outputGateBiases, lstm.hiddenOutputBiases].map { $0.copy() }
    
    lstm.apply(gradients: (weights, biases), learningRate: 1)
    
    let after = [lstm.forgetGateWeights, lstm.inputGateWeights, lstm.gateGateWeights,
                 lstm.outputGateWeights, lstm.hiddenOutputWeights]
    let afterBiases = [lstm.forgetGateBiases, lstm.inputGateBiases, lstm.gateGateBiases,
                       lstm.outputGateBiases, lstm.hiddenOutputBiases]
    
    for (b, a) in zip(before, after) {
      XCTAssertEqual(a.shape, b.shape)
      XCTAssertFalse(a.isValueEqual(to: b), "every weight group should change after apply")
    }
    for (b, a) in zip(beforeBiases, afterBiases) {
      XCTAssertEqual(a.shape, b.shape)
      XCTAssertFalse(a.isValueEqual(to: b), "every bias group should change after apply")
    }
  }
  
  func test_sameShapeVectors_addElementwise() {
    // (W,1,D) + (W,1,D) matches the "along rows" broadcast rule; the result must still be a
    // plain element-wise sum sized by storage.
    let size = TensorSize(rows: 1, columns: 8, depth: 3)
    let a = Tensor.fillRandom(size: size)
    let b = Tensor.fillRandom(size: size)
    let sum = a + b
    XCTAssertEqual(sum.storage.count, 24)
    for i in 0..<24 {
      XCTAssertEqual(sum.storage[i], a.storage[i] + b.storage[i], accuracy: 1e-6)
    }
  }
  
  func test_globalGradientNorm_matchesPerTensorSum() {
    let (embedding, lstm) = makeExplodingLSTM(recurrentErrorClip: 1.0)
    let embedded = embedding.forward(tensor: makeInput())
    let out = lstm.forward(tensor: embedded, context: .init())
    out.setGraph(embedded)
    let gradients = out.gradients(delta: Tensor.fillRandom(size: out.size), wrt: embedded)
    
    // note: `[Tensor] + [Tensor]` is element-wise in this codebase, so reduce each list separately
    let weightSumSq = gradients.weights.reduce(Tensor.Scalar(0)) { $0 + $1.storage.sumOfSquares }
    let biasSumSq = gradients.biases.reduce(Tensor.Scalar(0)) { $0 + $1.storage.sumOfSquares }
    let expected = Tensor.Scalar.sqrt(weightSumSq + biasSumSq + .stabilityFactor)
    
    XCTAssertEqual(gradients.calculateL2Norm(), expected, accuracy: expected * 1e-5)
  }
  
  // MARK: - Adam epsilon placement
  
  func test_Adam_tinyGradientStepIsNearLearningRate() {
    // With eps outside the sqrt, |step| -> lr as gradient -> 0 (standard Adam, t = 1).
    // With eps inside the sqrt (the old form), a 1e-10 gradient produced a step of ~1e-4 * lr.
    let network = Sequential {
      [Dense(3, inputs: 2, initializer: .heNormal, biasEnabled: false)]
    }
    
    let learningRate: Tensor.Scalar = 0.01
    let optimizer = Adam(network, learningRate: learningRate, batchSize: 1)
    
    let dense = network.layers[0]
    let before = dense.weights.copy()
    
    let gradient = Tensor.fillWith(value: 1e-10, size: dense.weights.size)
    let biasGradient = dense.biases.zerosLike()
    
    optimizer.zeroGradients()
    optimizer.apply(Tensor.Gradient(input: [], weights: [gradient], biases: [biasGradient]))
    optimizer.step()
    
    let after = dense.weights
    let step = before - after
    
    for value in step.asArray {
      XCTAssertEqual(value, learningRate, accuracy: learningRate * 0.02,
                     "each element should move by ~lr regardless of gradient scale")
    }
  }
  
  // MARK: - gradientNormInspector
  
  func test_gradientNormInspector_reportsPerGateNormsForLSTM() {
    let network = Sequential {
      [
        Embedding(inputUnits: inputUnits, vocabSize: vocabSize, batchLength: batchLength),
        LSTM(inputUnits: inputUnits,
             batchLength: batchLength,
             biasEnabled: true,
             hiddenUnits: hiddenUnits,
             vocabSize: vocabSize)
      ]
    }
    
    let optimizer = Adam(network, learningRate: 0.001, batchSize: 1)
    
    var received: [GradientNormReport] = []
    optimizer.gradientNormInspector = { received = $0 }
    
    let label = Tensor.fillRandom(size: network.layers.last!.outputSize)
    
    optimizer.zeroGradients()
    let output = optimizer.fit([makeInput()], labels: [label], lossFunction: .meanSquareError)
    optimizer.apply(output.gradients)
    optimizer.step()
    
    let lstmReports = received.filter { $0.layer == "LSTM" }
    XCTAssertEqual(lstmReports.map(\.group),
                   ["forgetGate", "inputGate", "gateGate", "outputGate", "hiddenOutput", "(all)"])
    XCTAssertEqual(Set(lstmReports.map(\.layerIndex)), [1])
    
    for report in lstmReports {
      XCTAssertTrue(report.weightNorm.isFinite)
      XCTAssertGreaterThan(report.weightNorm, 0, "\(report.group) weight gradient should be non-zero")
    }
    
    // "(all)" is the whole packed tensor; it must be fully accounted for by the groups.
    let groups = lstmReports.filter { $0.group != "(all)" }
    let all = try! XCTUnwrap(lstmReports.first { $0.group == "(all)" })
    let groupWeightNorm = Tensor.Scalar.sqrt(groups.reduce(0) { $0 + $1.weightNorm * $1.weightNorm })
    let groupBiasNorm = Tensor.Scalar.sqrt(groups.reduce(0) { $0 + $1.biasNorm * $1.biasNorm })
    XCTAssertEqual(all.weightNorm, groupWeightNorm, accuracy: groupWeightNorm * 1e-4)
    XCTAssertEqual(all.biasNorm, groupBiasNorm, accuracy: groupBiasNorm * 1e-4)
    
    let embeddingReports = received.filter { $0.layer == "Embedding" }
    XCTAssertEqual(embeddingReports.count, 1)
    XCTAssertEqual(embeddingReports.first?.group, "weights")
  }
  
  func test_gradientNormInspector_notCalledWhenUnset() {
    let network = Sequential {
      [Dense(3, inputs: 2, initializer: .heNormal, biasEnabled: false)]
    }
    let optimizer = Adam(network, learningRate: 0.01, batchSize: 1)
    XCTAssertNil(optimizer.gradientNormInspector)
    
    // Must not crash or require a reporter when the hook is unset.
    optimizer.zeroGradients()
    optimizer.apply(Tensor.Gradient(input: [],
                                    weights: [Tensor.fillWith(value: 0.1, size: network.layers[0].weights.size)],
                                    biases: [network.layers[0].biases.zerosLike()]))
    optimizer.step()
  }
}
