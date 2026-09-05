@testable import Neuron
import XCTest

final class LSTMApplyMappingTests: XCTestCase {
  /// Packs five distinct constant tensors in `weights`/`biases` order and checks that `apply`
  /// subtracts each one from the matching parameter (i.e. the flat layout round-trips).
  func test_apply_mapsEachPackedGroupToItsParameter() {
    let lstm = LSTM(inputUnits: 4, batchLength: 3, biasEnabled: true, hiddenUnits: 8, vocabSize: 5)
    
    let wShapes = [lstm.forgetGateWeights, lstm.inputGateWeights, lstm.gateGateWeights, lstm.outputGateWeights, lstm.hiddenOutputWeights]
    let bShapes = [lstm.forgetGateBiases, lstm.inputGateBiases, lstm.gateGateBiases, lstm.outputGateBiases, lstm.hiddenOutputBiases]
    let wBefore = wShapes.map { $0.copy() }
    let bBefore = bShapes.map { $0.copy() }
    
    let wGroups = wShapes.enumerated().map { Tensor.fillWith(value: Tensor.Scalar($0.offset + 1), size: $0.element.size) }
    let bGroups = bShapes.enumerated().map { Tensor.fillWith(value: Tensor.Scalar(10 * ($0.offset + 1)), size: $0.element.size) }
    let packedW = wGroups.dropFirst().reduce(wGroups[0]) { $0.concat($1, axis: -1) }
    let packedB = bGroups.dropFirst().reduce(bGroups[0]) { $0.concat($1, axis: -1) }
    
    XCTAssertEqual(packedW.storage.count, lstm.weights.storage.count)
    XCTAssertEqual(packedB.storage.count, lstm.biases.storage.count)
    
    lstm.apply(gradients: (packedW, packedB), learningRate: 1)
    
    let wAfter = [lstm.forgetGateWeights, lstm.inputGateWeights, lstm.gateGateWeights, lstm.outputGateWeights, lstm.hiddenOutputWeights]
    let bAfter = [lstm.forgetGateBiases, lstm.inputGateBiases, lstm.gateGateBiases, lstm.outputGateBiases, lstm.hiddenOutputBiases]
    
    for i in 0..<5 {
      let expectedW = wBefore[i] - Tensor.Scalar(i + 1)
      XCTAssertTrue(wAfter[i].isValueEqual(to: expectedW), "weight group \(i) got the wrong gradient")
      let expectedB = bBefore[i] - Tensor.Scalar(10 * (i + 1))
      XCTAssertTrue(bAfter[i].isValueEqual(to: expectedB), "bias group \(i) got the wrong gradient")
    }
  }
}
