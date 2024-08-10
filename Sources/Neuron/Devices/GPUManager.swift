import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate
import NumSwift

// MARK: WIP - This is a work in progress and is not ready for use.

class GPUManager {
  enum MetalFunction: String {
    case activation, derivate
  }
  
  private let numberOfConcurrentBuffers: Int = 64

  private lazy var device: MTLDevice? = {
    return MTLCreateSystemDefaultDevice()
  }()
  
  private lazy var commandQueue: MTLCommandQueue? = {
    if let device = self.device {
      let queue = device.makeCommandQueue(maxCommandBufferCount: numberOfConcurrentBuffers)
      return queue
    }
    
    return nil
  }()
  
  private lazy var defaultLibrary = try? device?.makeDefaultLibrary(bundle: Bundle.module)
  
  /* TODO:
     Create a queue of commands,
     wait until all commands are done by label (maybe use a semaphore or something?)
     allow commands to go through
  */
  
  func matmul(_ A: [[Tensor.Scalar]],
              _ aShape: [Int],
              _ B: [[Tensor.Scalar]],
              _ bShape: [Int]) -> [[Tensor.Scalar]] {
    
    let bColumns = bShape[safe: 0] ?? 0
    let bRows = bShape[safe: 1] ?? 0
    
    let aColumns = aShape[safe: 0] ?? 0
    let aRows = aShape[safe: 1] ?? 0
    
    var M = Int32(aRows)
    var K = Int32(aColumns)
    var N = Int32(bColumns)
    
    guard K == Int32(bRows) else {
      print("Matrix dimensions don't match for multiplication")
      return []
    }
    
    // Flatten 2D arrays
    let flatA = A.flatten()
    let flatB = B.flatten()
    
    guard let device,
          let defaultLibrary,
          let function = defaultLibrary.makeFunction(name: "matmul"),
          let commandQueue,
          let commandBuffer = commandQueue.makeCommandBuffer(),
          let computeEncoder = commandBuffer.makeComputeCommandEncoder()
    else {
      fatalError("Failed to create Metal function")
      return []
    }
    
    commandBuffer.label = "matmul"

    let pipelineState: MTLComputePipelineState
    do {
      pipelineState = try device.makeComputePipelineState(function: function)
    } catch {
      fatalError("Failed to create compute pipeline state: \(error)")
      return []
    }
    
    let aBuffer = device.makeBuffer(bytes: flatA, length: flatA.count * MemoryLayout<Float>.stride, options: [])
    let bBuffer = device.makeBuffer(bytes: flatB, length: flatB.count * MemoryLayout<Float>.stride, options: [])
    guard let cBuffer = device.makeBuffer(length: Int(M * N) * MemoryLayout<Float>.stride, options: []) else { return [] }
    
    computeEncoder.setComputePipelineState(pipelineState)
    computeEncoder.setBuffer(aBuffer, offset: 0, index: 0)
    computeEncoder.setBuffer(bBuffer, offset: 0, index: 1)
    computeEncoder.setBuffer(cBuffer, offset: 0, index: 2)
    computeEncoder.setBytes(&M, length: MemoryLayout<Int32>.stride, index: 3)
    computeEncoder.setBytes(&N, length: MemoryLayout<Int32>.stride, index: 4)
    computeEncoder.setBytes(&K, length: MemoryLayout<Int32>.stride, index: 5)
    
    let gridSize = MTLSizeMake(Int(N), Int(M), 1)
    let threadGroupSize = MTLSizeMake(16, 16, 1)
    computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    
    computeEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let resultData = Data(bytesNoCopy: cBuffer.contents(), count: cBuffer.length, deallocator: .none)
    let resultArray = resultData.withUnsafeBytes { Array(UnsafeBufferPointer<Float>(start: $0.baseAddress!.assumingMemoryBound(to: Float.self), count: Int(M * N))) }
    
    return resultArray.reshape(columns: Int(N))
  }
  
  func conv2d(input: [[Tensor.Scalar]],
              kernels: [[Tensor.Scalar]],
              strides: (Int, Int) = (1,1),
              padding: NumSwift.ConvPadding = .valid,
              filterSize: (rows: Int, columns: Int),
              inputSize: (rows: Int, columns: Int),
              outputSize: (rows: Int, columns: Int)? = nil,
              transConv: Bool = false) -> [[Float]] {
    
    let padding = padding.extra(inputSize: inputSize, filterSize: filterSize, stride: strides)
    
    let function = transConv ? "transConv2d" : "conv2d"
    
    guard let device,
          let defaultLibrary,
          let kernelFunction = defaultLibrary.makeFunction(name: function),
          let pipelineState = try? device.makeComputePipelineState(function: kernelFunction),
          let commandQueue,
          let commandBuffer = commandQueue.makeCommandBuffer(),
          let computeEncoder = commandBuffer.makeComputeCommandEncoder()
    else {
      fatalError("Failed to create Metal function")
      return []
    }
    
    commandBuffer.label = function
    
    var inputHeight = Int32(input.count)
    var inputWidth = Int32(inputSize.columns)
    let inputChannels = 1
    var kernelSize = Int32(filterSize.columns)
    
    var outputWidth = outputSize?.columns ?? 1
    var outputHeight = outputSize?.rows ?? 1
    let outputChannels = 1
    
    // Flatten 2D arrays
    let flatInput = input.flatten()
    let flatKernels = kernels.flatten()
    
    let inputBuffer = device.makeBuffer(bytes: flatInput, length: flatInput.count * MemoryLayout<Float>.stride, options: [])
    let kernelBuffer = device.makeBuffer(bytes: flatKernels, length: flatKernels.count * MemoryLayout<Float>.stride, options: [])
    guard let outputBuffer = device.makeBuffer(length: Int(outputWidth * outputHeight * outputChannels) * MemoryLayout<Float>.stride, options: []) else { return [] }
        
    computeEncoder.setComputePipelineState(pipelineState)
    computeEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
    computeEncoder.setBuffer(kernelBuffer, offset: 0, index: 1)
    computeEncoder.setBuffer(outputBuffer, offset: 0, index: 2)
    computeEncoder.setBytes(&inputWidth, length: MemoryLayout<Int32>.stride, index: 3)
    computeEncoder.setBytes(&inputHeight, length: MemoryLayout<Int32>.stride, index: 4)
    var inputChannels32 = Int32(inputChannels)
    computeEncoder.setBytes(&inputChannels32, length: MemoryLayout<Int32>.stride, index: 5)
    computeEncoder.setBytes(&kernelSize, length: MemoryLayout<Int32>.stride, index: 6)
    computeEncoder.setBytes(&outputWidth, length: MemoryLayout<Int32>.stride, index: 7)
    computeEncoder.setBytes(&outputHeight, length: MemoryLayout<Int32>.stride, index: 8)
    var outputChannels32 = Int32(outputChannels)
    computeEncoder.setBytes(&outputChannels32, length: MemoryLayout<Int32>.stride, index: 9)
    var strideX = Int32(strides.1)
    var strideY = Int32(strides.0)
    computeEncoder.setBytes(&strideX, length: MemoryLayout<Int32>.stride, index: 10)
    computeEncoder.setBytes(&strideY, length: MemoryLayout<Int32>.stride, index: 11)
    var paddingX = Int32(padding.left)
    var paddingY = Int32(padding.top)
    computeEncoder.setBytes(&paddingX, length: MemoryLayout<Int32>.stride, index: 12)
    computeEncoder.setBytes(&paddingY, length: MemoryLayout<Int32>.stride, index: 13)
    
    let gridSize = MTLSizeMake(Int(outputWidth), Int(outputHeight), outputChannels)
    let threadGroupSize = MTLSizeMake(16, 16, 1)
    computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    
    computeEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let outputData = Data(bytesNoCopy: outputBuffer.contents(), count: outputBuffer.length, deallocator: .none)
    let flatResult = [Float](unsafeUninitializedCapacity: outputData.count / MemoryLayout<Float>.stride) { buffer, initializedCount in
      outputData.copyBytes(to: buffer)
      initializedCount = outputData.count / MemoryLayout<Float>.stride
    }
    
    return flatResult.reshape(columns: outputWidth)
  }
  
  func activate(to input: [[Tensor.Scalar]],
                inputSize: (rows: Int, columns: Int),
                activationType: Activation,
                derivate: Bool = false) -> [[Float]] {
    var height = UInt32(inputSize.rows)
    var width = UInt32(inputSize.columns)
    
    // Flatten 2D array
    let flatInput = input.flatten()
    let function: MetalFunction = derivate ? .derivate : .activation
    
    guard let device,
          let defaultLibrary,
          let kernelFunction = defaultLibrary.makeFunction(name: function.rawValue),
          let pipelineState = try? device.makeComputePipelineState(function: kernelFunction),
          let commandQueue,
          let commandBuffer = commandQueue.makeCommandBuffer(),
          let computeEncoder = commandBuffer.makeComputeCommandEncoder()
    else {
      fatalError("Failed to create Metal function")
      return []
    }
    
    let label = "activate-\(activationType.asString())-\(derivate)"
    commandBuffer.label = label

    let inputBuffer = device.makeBuffer(bytes: flatInput, length: flatInput.count * MemoryLayout<Float>.stride, options: [])
    let outputBuffer = device.makeBuffer(length: flatInput.count * MemoryLayout<Float>.stride, options: [])
        
    computeEncoder.setComputePipelineState(pipelineState)
    computeEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
    computeEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
    
    var activationTypeRaw = activationType.index()
    computeEncoder.setBytes(&activationTypeRaw, length: MemoryLayout<UInt32>.stride, index: 2)
    
    var defaultLimit: Int = 0
    switch activationType {
    case .leakyRelu(let limit):
      var limit = Tensor.Scalar(limit)
      computeEncoder.setBytes(&limit, length: MemoryLayout<Tensor.Scalar>.size, index: 3)
    default:
      computeEncoder.setBytes(&defaultLimit, length: MemoryLayout<Tensor.Scalar>.size, index: 3)
      break
    }
    
    computeEncoder.setBytes(&width, length: MemoryLayout<UInt32>.stride, index: 4)
    computeEncoder.setBytes(&height, length: MemoryLayout<UInt32>.stride, index: 5)
    
    let gridSize = MTLSizeMake(Int(width), Int(height), 1)
    let threadGroupSize = MTLSizeMake(16, 16, 1)
    computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    
    computeEncoder.endEncoding()
    commandBuffer.commit()
    
    commandBuffer.addCompletedHandler { buffer in
      
    }
    
    commandBuffer.waitUntilCompleted()
    
    let resultData = Data(bytesNoCopy: outputBuffer!.contents(), count: outputBuffer!.length, deallocator: .none)
    let resultArray = resultData.withUnsafeBytes { Array(UnsafeBufferPointer<Float>(start: $0.baseAddress!.assumingMemoryBound(to: Float.self), count: flatInput.count)) }
    
    return resultArray.reshape(columns: Int(width))
  }
}

