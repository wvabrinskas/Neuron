import XCTest
@testable import Neuron
import Metal

final class MetalEngineTests: XCTestCase {

  func testMetalEngineIsAvailable() throws {
    try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal not available")
    let engine = MetalEngine()
    XCTAssertTrue(engine.isAvailable)
    XCTAssertNotNil(engine.device)
    XCTAssertNotNil(engine.commandQueue)
    XCTAssertNotNil(engine.bufferPool)
  }

  func testMetalEnginePipelineLoading() throws {
    try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal not available")
    let engine = MetalEngine()
    // NeuronKernels.metal has "neuron_activation" kernel
    // Skip if default library unavailable (e.g. SPM without Metal compilation)
    try XCTSkipIf(engine.device?.makeDefaultLibrary() == nil, "Default Metal library not available")
    let pipeline = engine.pipeline(named: "neuron_activation")
    XCTAssertNotNil(pipeline, "Should load neuron_activation pipeline from default library")
  }

  func testMetalEngineAcquireReleaseBuffer() throws {
    try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal not available")
    let engine = MetalEngine()
    let buf = engine.acquireBuffer(byteCount: 256)
    XCTAssertNotNil(buf)
    XCTAssertGreaterThanOrEqual(buf!.length, 256)
    engine.releaseBuffer(buf!)
  }

  func testMetalEngineMakeCommandBuffer() throws {
    try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal not available")
    let engine = MetalEngine()
    let cmdBuffer = engine.makeCommandBuffer()
    XCTAssertNotNil(cmdBuffer)
  }
}
