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

  func testGPUActivateWithMetalTensorStorageMatchesCPU() throws {
    try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal not available")
    try XCTSkipIf(MetalContext.shared.device?.makeDefaultLibrary() == nil, "Default Metal library not available")
    let device = MetalContext.shared.device!
    let pool = MetalContext.shared.bufferPool
    let data: [Tensor.Scalar] = [-2, -1, 0, 1, 2, 3]
    let metalStorage = MetalTensorStorage(device: device, data: data, pool: pool)
    let metalInput = Tensor(storage: metalStorage, size: TensorSize(rows: 1, columns: data.count, depth: 1), context: TensorContext())
    let cpuInput = Tensor(data, context: TensorContext())
    let gpu = GPU()
    let cpu = CPU()
    let metalResult = gpu.activate(metalInput, .reLu)
    let cpuResult = cpu.activate(cpuInput, .reLu)
    let metalArray = metalResult.storage.toArray()
    let cpuArray = cpuResult.storage.toArray()
    XCTAssertEqual(metalArray.count, cpuArray.count)
    for i in 0..<metalArray.count {
      XCTAssertEqual(metalArray[i], cpuArray[i], accuracy: 1e-5, "Index \(i)")
    }
  }

  func testGPUDerivateWithMetalTensorStorageMatchesCPU() throws {
    try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal not available")
    try XCTSkipIf(MetalContext.shared.device?.makeDefaultLibrary() == nil, "Default Metal library not available")
    let device = MetalContext.shared.device!
    let pool = MetalContext.shared.bufferPool
    let data: [Tensor.Scalar] = [-2, -1, 0, 1, 2, 3]
    let metalStorage = MetalTensorStorage(device: device, data: data, pool: pool)
    let metalInput = Tensor(storage: metalStorage, size: TensorSize(rows: 1, columns: data.count, depth: 1), context: TensorContext())
    let cpuInput = Tensor(data, context: TensorContext())
    let gpu = GPU()
    let cpu = CPU()
    let metalResult = gpu.derivate(metalInput, .reLu)
    let cpuResult = cpu.derivate(cpuInput, .reLu)
    let metalArray = metalResult.storage.toArray()
    let cpuArray = cpuResult.storage.toArray()
    XCTAssertEqual(metalArray.count, cpuArray.count)
    for i in 0..<metalArray.count {
      XCTAssertEqual(metalArray[i], cpuArray[i], accuracy: 1e-5, "Index \(i)")
    }
  }

  func testGPUMatmulWithMetalTensorStorageMatchesCPU() throws {
    try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal not available")
    try XCTSkipIf(MetalContext.shared.device?.makeDefaultLibrary() == nil, "Default Metal library not available")
    let device = MetalContext.shared.device!
    let pool = MetalContext.shared.bufferPool

    // A: 2×3, B: 3×4 → C: 2×4
    let dataA: [Tensor.Scalar] = [1, 2, 3, 4, 5, 6]
    let dataB: [Tensor.Scalar] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    let metalStorageA = MetalTensorStorage(device: device, data: dataA, pool: pool)
    let metalStorageB = MetalTensorStorage(device: device, data: dataB, pool: pool)
    let metalA = Tensor(storage: metalStorageA, size: TensorSize(rows: 2, columns: 3, depth: 1), context: TensorContext())
    let metalB = Tensor(storage: metalStorageB, size: TensorSize(rows: 3, columns: 4, depth: 1), context: TensorContext())

    let cpuA = Tensor([[1, 2, 3], [4, 5, 6]], context: TensorContext())
    let cpuB = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], context: TensorContext())

    let gpu = GPU()
    let cpu = CPU()
    let metalResult = gpu.matmul(metalA, metalB)
    let cpuResult = cpu.matmul(cpuA, cpuB)

    let metalArray = metalResult.storage.toArray()
    let cpuArray = cpuResult.storage.toArray()
    XCTAssertEqual(metalArray.count, cpuArray.count)
    for i in 0..<metalArray.count {
      XCTAssertEqual(metalArray[i], cpuArray[i], accuracy: 1e-5, "Index \(i)")
    }
  }

  func testGPUMatmulDepth2MatchesCPU() throws {
    try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal not available")
    try XCTSkipIf(MetalContext.shared.device?.makeDefaultLibrary() == nil, "Default Metal library not available")
    let device = MetalContext.shared.device!
    let pool = MetalContext.shared.bufferPool

    // A: 2×3×2, B: 3×4×2 → C: 2×4×2 (depth=2)
    let dataA: [Tensor.Scalar] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    let dataB: [Tensor.Scalar] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

    let metalStorageA = MetalTensorStorage(device: device, data: dataA, pool: pool)
    let metalStorageB = MetalTensorStorage(device: device, data: dataB, pool: pool)
    let metalA = Tensor(storage: metalStorageA, size: TensorSize(rows: 2, columns: 3, depth: 2), context: TensorContext())
    let metalB = Tensor(storage: metalStorageB, size: TensorSize(rows: 3, columns: 4, depth: 2), context: TensorContext())

    let cpuA = Tensor(storage: TensorStorage.create(from: dataA), size: TensorSize(rows: 2, columns: 3, depth: 2), context: TensorContext())
    let cpuB = Tensor(storage: TensorStorage.create(from: dataB), size: TensorSize(rows: 3, columns: 4, depth: 2), context: TensorContext())

    let gpu = GPU()
    let cpu = CPU()
    let metalResult = gpu.matmul(metalA, metalB)
    let cpuResult = cpu.matmul(cpuA, cpuB)

    let metalArray = metalResult.storage.toArray()
    let cpuArray = cpuResult.storage.toArray()
    XCTAssertEqual(metalArray.count, cpuArray.count)
    for i in 0..<metalArray.count {
      XCTAssertEqual(metalArray[i], cpuArray[i], accuracy: 1e-5, "Index \(i)")
    }
  }
}
