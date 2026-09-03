@testable import Neuron
import XCTest

/// Central-difference check of the LSTM's analytic weight gradients under the loss the layer is
/// designed for (`crossEntropySoftmax`, whose derivative already folds in the softmax Jacobian).
final class LSTMNumericalGradientTests: XCTestCase {
  
  private struct GroupResult { let name: String; let maxRel: Tensor.Scalar; let worst: String; let medianRatio: Tensor.Scalar }
  
  private func check(steps: Int, gateScale: Tensor.Scalar) -> [GroupResult] {
    let inputUnits = 3, hidden = 4, vocab = 5
    let lstm = LSTM(inputUnits: inputUnits, batchLength: steps, returnSequence: true, biasEnabled: true,
                    initializer: .xavierNormal, hiddenUnits: hidden, vocabSize: vocab)
    for kp in [\LSTM.forgetGateWeights, \LSTM.inputGateWeights, \LSTM.gateGateWeights, \LSTM.outputGateWeights] {
      lstm[keyPath: kp] = Tensor(storage: lstm[keyPath: kp].storage * gateScale, size: lstm[keyPath: kp].size)
    }
    let input = Tensor.fillRandom(in: -1...1, size: TensorSize(rows: 1, columns: inputUnits, depth: steps))
    var labelValues: [[[Tensor.Scalar]]] = []
    for t in 0..<steps { var oh = [Tensor.Scalar](repeating: 0, count: vocab); oh[(t * 2 + 1) % vocab] = 1; labelValues.append([oh]) }
    let label = Tensor(labelValues)
    let lossFn = LossFunction.crossEntropySoftmax
    
    func loss() -> Tensor.Scalar {
      lossFn.calculate(lstm.forward(tensor: input, context: .init()), correct: label).sum(axis: -1).asScalar()
    }
    let out = lstm.forward(tensor: input, context: .init())
    out.setGraph(input)
    let delta = lossFn.derivative(out, correct: label)
    let gradients = out.gradients(delta: delta, wrt: input)
    let analyticWeights = gradients.weights.first(where: { !$0.isEmpty })!
    let analyticBiases = gradients.biases.first(where: { !$0.isEmpty })!
    
    let groups: [(String, WritableKeyPath<LSTM, Tensor>, Tensor)] = [
      ("forgetGate", \LSTM.forgetGateWeights, analyticWeights), ("inputGate", \LSTM.inputGateWeights, analyticWeights),
      ("gateGate", \LSTM.gateGateWeights, analyticWeights), ("outputGate", \LSTM.outputGateWeights, analyticWeights),
      ("hiddenOutput", \LSTM.hiddenOutputWeights, analyticWeights),
      ("forgetBias", \LSTM.forgetGateBiases, analyticBiases), ("inputBias", \LSTM.inputGateBiases, analyticBiases),
      ("gateBias", \LSTM.gateGateBiases, analyticBiases), ("outputBias", \LSTM.outputGateBiases, analyticBiases),
      ("hiddenOutputBias", \LSTM.hiddenOutputBiases, analyticBiases)]
    var results: [GroupResult] = []
    var offset = 0
    let h: Tensor.Scalar = 1e-2
    for (name, kp, analytic) in groups {
      if name == "forgetBias" { offset = 0 } // bias groups are packed in their own tensor
      let count = lstm[keyPath: kp].storage.count
      var numerics: [Tensor.Scalar] = []
      for i in 0..<count {
        let original = lstm[keyPath: kp].storage[i]
        lstm[keyPath: kp].storage[i] = original + h; let plus = loss()
        lstm[keyPath: kp].storage[i] = original - h; let minus = loss()
        lstm[keyPath: kp].storage[i] = original
        numerics.append((plus - minus) / (2 * h))
      }
      // Errors are measured relative to the group's largest gradient so that entries that are
      // legitimately ~0 (e.g. hidden-state rows at t=0) are judged on the group's scale.
      let scale = max(numerics.map { abs($0) }.max() ?? 0, 1e-4)
      var maxRel: Tensor.Scalar = 0, worst = ""
      var ratios: [Tensor.Scalar] = []
      for i in 0..<count {
        let numeric = numerics[i]
        let a = analytic.storage[offset + i]
        let rel = abs(a - numeric) / scale
        if abs(numeric) > 0.2 * scale { ratios.append(a / numeric) }
        if rel > maxRel { maxRel = rel; worst = String(format: "analytic %.4f vs numeric %.4f (group scale %.4f)", a, numeric, scale) }
      }
      ratios.sort()
      results.append(GroupResult(name: name, maxRel: maxRel, worst: worst, medianRatio: ratios.isEmpty ? .nan : ratios[ratios.count / 2]))
      offset += count
    }
    return results
  }
  
  func test_LSTM_weightGradients_matchFiniteDifferences() {
    for steps in [1, 3, 6] {
      let results = check(steps: steps, gateScale: 3)
      print("FD steps=\(steps): " + results.map { String(format: "%@ maxRel %.3f ratio %.2f", $0.name, $0.maxRel, $0.medianRatio) }.joined(separator: " | "))
      for r in results {
        XCTAssertLessThan(r.maxRel, 0.05, "steps=\(steps) \(r.name): \(r.worst)")
      }
    }
  }
}
