import XCTest
@testable import Neuron
import Metal

final class MetalTensorStorageIntegrationTests: XCTestCase {

  func testTensorsHaveMTLBufferWhenMetalAvailable() throws {
    try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal not available")
    let t = Tensor([1.0, 2.0, 3.0])
    let metalStorage = t.storage as? MetalTensorStorage
    XCTAssertNotNil(metalStorage)
    XCTAssertNotNil(metalStorage?.mtlBuffer)
  }

  func testForwardBackwardPassWithMetalBackedTensors() throws {
    try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal not available")
    let network = Sequential {
      [
        Dense(8, inputs: 4),
        ReLu(),
        Dense(2),
        Softmax()
      ]
    }
    network.compile()
    let optimizer = Adam(network, learningRate: 0.01, batchSize: 1)
    let input = Tensor([[1.0, 2.0, 3.0, 4.0]])
    let label = Tensor([[0.0, 1.0]])
    optimizer.zeroGradients()
    let (_, gradients, loss, _) = optimizer.fit(
      [input],
      labels: [label],
      lossFunction: .crossEntropySoftmax
    )
    optimizer.apply(gradients)
    optimizer.step()
    XCTAssertFalse(loss.isNaN)
    XCTAssertGreaterThan(loss, 0)
  }

  func testSerializationDeserializationMetalAndCPU() throws {
    try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal not available")
    let original = Tensor([[1.0, 2.0], [3.0, 4.0]])
    let encoder = JSONEncoder()
    let data = try encoder.encode(original)
    let decoder = JSONDecoder()
    let decoded = try decoder.decode(Tensor.self, from: data)
    XCTAssertEqual(original.storage.toArray(), decoded.storage.toArray())
    XCTAssertEqual(original.size, decoded.size)
  }
}
