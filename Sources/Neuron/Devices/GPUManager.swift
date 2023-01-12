import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate
import NumSwift

extension MTLTexture {
  func getValues(device: MTLDevice, region: MTLRegion? = nil) -> Tensor {
    let region = region ?? MTLRegionMake2D(0, 0, self.width, self.height)
    let rowBytes = region.size.width * MemoryLayout<Float>.stride
    let byteCount = region.size.height * rowBytes
    let totalSize = region.size.width * region.size.height
    guard let buffer = device.makeBuffer(length: byteCount, options: []) else { return Tensor() }
    getBytes(buffer.contents(), bytesPerRow: rowBytes, from: region, mipmapLevel: 0)
    let values = Array(UnsafeBufferPointer(start: buffer.contents().bindMemory(to: Float.self, capacity: byteCount), count: totalSize))
    return Tensor(values.reshape(columns: region.size.width))
  }
}

public typealias DataType = [[CFloat]]
public typealias ResultType = CFloat

public class GPUManager {
  public enum MetalFunction: String {
    case activation, derivate, conv2d
  }
  
  private var currentRunningPipelines: [MTLComputePipelineState] = []
  private var device: MTLDevice? = MTLCreateSystemDefaultDevice()
  
  @Atomic
  private var textures: [Int: MTLTexture] = [:]
  
  lazy var queue = self.device?.makeCommandQueue()
  lazy var cmds = queue?.makeCommandBuffer()
  
  private func getFunction(_ function: MetalFunction) -> MTLFunction? {
    return try? device?.makeDefaultLibrary(bundle: Bundle.module).makeFunction(name: function.rawValue)
  }
  
  private func pipelineIfExists(type: MetalFunction) -> MTLComputePipelineState? {
    return self.currentRunningPipelines.filter({ $0.label == type.rawValue }).first
  }
  
  private func addPipeline(for type: MetalFunction) -> MTLComputePipelineState? {
    guard let device = self.device,
          let function = getFunction(type) else {
      return nil
    }
    
    do {
      let descriptor = MTLComputePipelineDescriptor()
      descriptor.label = type.rawValue
      descriptor.computeFunction = function
      
      let pipeline = try device.makeComputePipelineState(descriptor: descriptor,
                                                         options: [],
                                                         reflection: nil)
      
      self.currentRunningPipelines.append(pipeline)
      return pipeline
      
    } catch {
      print(error)
      return nil
    }
  }
  
  public func commit() -> [Tensor] {
    cmds?.commit()
    cmds?.waitUntilCompleted()
    return []
  }
  
  private func conv2dOutputSize(padding: NumSwift.ConvPadding,
                                strides: (rows: Int, columns: Int),
                                filterCount: Int,
                                filterSize: (rows: Int, columns: Int),
                                inputSize: (rows: Int, columns: Int, depth: Int)) -> TensorSize {
    let paddingValue = padding.extra(inputSize: (inputSize.rows, inputSize.columns), filterSize: filterSize)
    
    let rows = (((inputSize.rows + (paddingValue.top + paddingValue.bottom)) - (filterSize.rows - 1) - 1) / strides.rows) + 1
    let columns = (((inputSize.columns + (paddingValue.left + paddingValue.right)) - (filterSize.columns - 1) - 1) / strides.columns) + 1
    
    return TensorSize(array: [columns, rows, filterCount])
  }
  
  public func conv2d(_ input: Tensor,
                     filters: [Tensor],
                     biases: Tensor,
                     padding: NumSwift.ConvPadding,
                     filterSize: (rows: Int, columns: Int),
                     strides: (rows: Int, columns: Int),
                     inputSize: (rows: Int, columns: Int, depth: Int)) -> [Tensor] {
    
    let outputSize = conv2dOutputSize(padding: padding,
                                      strides: strides,
                                      filterCount: filters.count,
                                      filterSize: filterSize,
                                      inputSize: inputSize)
    
    guard let device = device else {
      return []
    }
    
    let function: MetalFunction = .conv2d
    let pipeline: MTLComputePipelineState? = self.pipelineIfExists(type: function) ?? self.addPipeline(for: function)
    
    // input texture
    let inputTextureDesc = MTLTextureDescriptor()
    inputTextureDesc.textureType = .type2D
    inputTextureDesc.width = inputSize.columns
    inputTextureDesc.height = inputSize.rows
    inputTextureDesc.pixelFormat = .r32Float
    inputTextureDesc.usage = .shaderRead
    
    guard let inputTexture = device.makeTexture(descriptor: inputTextureDesc) else { return [] }
    
    let flatInput: [Float] = input.value.flatten()
    let pointer = UnsafeMutableRawPointer(mutating: flatInput)
    let bytesPerRow = inputTexture.width * MemoryLayout<Float>.stride
    let region = MTLRegionMake2D(0, 0, inputTexture.width, inputTexture.height)
    inputTexture.replace(region: region, mipmapLevel: 0, withBytes: pointer, bytesPerRow: bytesPerRow)
    
    let newEncoder = cmds?.makeComputeCommandEncoder()
    
    guard let encoder = newEncoder, let pipelineStrong = pipeline else {
      return []
    }
    
    encoder.setComputePipelineState(pipelineStrong)
  
    let w = pipelineStrong.threadExecutionWidth
    let h = pipelineStrong.maxTotalThreadsPerThreadgroup / w
    let threadsPerThreadgroup = MTLSizeMake(w, h, 1)
    
    let threadgroupsPerGrid = MTLSize(width: outputSize.columns,
                                      height: outputSize.rows,
                                      depth: outputSize.depth)
    
    let outputTextureDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float,
                                                                     width: outputSize.columns,
                                                                     height: outputSize.rows,
                                                                     mipmapped: false)
    
    encoder.setTexture(inputTexture, index: 0)
    
    var inSize = SIMD2(CUnsignedInt(inputSize.rows), CUnsignedInt(inputSize.columns))
    var outSize = SIMD2(CUnsignedInt(outputSize.rows), CUnsignedInt(outputSize.columns))
    var kSize = SIMD2(CUnsignedInt(filterSize.rows), CUnsignedInt(filterSize.columns))
    var strides = SIMD2(CUnsignedInt(strides.rows), CUnsignedInt(strides.columns))
    var padding = padding == .same ? 1 : 0
    
    encoder.setBytes(&inSize, length: MemoryLayout<SIMD2<CUnsignedInt>>.size, index: 1)
    encoder.setBytes(&outSize, length: MemoryLayout<SIMD2<CUnsignedInt>>.size, index: 2)
    encoder.setBytes(&kSize, length: MemoryLayout<SIMD2<CUnsignedInt>>.size, index: 3)
    encoder.setBytes(&strides, length: MemoryLayout<SIMD2<CUnsignedInt>>.size, index: 4)
    encoder.setBytes(&padding, length: MemoryLayout<Int>.size, index: 5)
    
    var outputTextures: [MTLTexture] = []
    
    filters.forEach { filter in
      // output texture
      guard let outputTexture = device.makeTexture(descriptor: outputTextureDesc) else { return }
      
      var filtersFlat: [Float] = filter.value.flatten()
      
      guard let filterBuffer = device.makeBuffer(bytes: &filtersFlat,
                                                 length: MemoryLayout<Float>.stride * filtersFlat.count,
                                                 options: []) else {
        return 
      }
      
      encoder.setTexture(outputTexture, index: 1)
      encoder.setBuffer(filterBuffer, offset: 0, index: 0)
      
      outputTextures.append(outputTexture)
    }
    
    encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    cmds?.commit()
    cmds?.waitUntilCompleted()
    
    // this is slow
    return outputTextures.map { $0.getValues(device: device) }
  }
  
  public func activate(_ num: [Float],
                       _ activationType: Activation,
                       derivate: Bool = false) -> [Float] {
    var data = num
    
    guard let device = self.device else {
      return num
    }
    
    let function: MetalFunction = derivate ? .derivate : .activation
    
    let pipeline: MTLComputePipelineState? = self.pipelineIfExists(type: function) ?? self.addPipeline(for: function)
    
    guard let dataBuffer = device.makeBuffer(bytes: &data,
                                             length: MemoryLayout<Float>.stride * data.count,
                                             options: []),
          
            let resultsBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * data.count,
                                                  options: []) else {
      return num
    }
    
    let newEncoder = cmds?.makeComputeCommandEncoder()
    
    guard let encoder = newEncoder, let pipelineStrong = pipeline else {
      return num
    }
    
    var activation = CUnsignedInt(activationType.index())
    
    encoder.setComputePipelineState(pipelineStrong)
    
    encoder.setBuffer(dataBuffer, offset: 0, index: 0)
    encoder.setBuffer(resultsBuffer, offset: 0, index: 1)
    encoder.setBytes(&activation, length: MemoryLayout<CUnsignedInt>.size, index: 2)
    
    switch activationType {
    case .leakyRelu(let limit):
      var limit = Float(limit)
      encoder.setBytes(&limit, length: MemoryLayout<Float>.size, index: 3)
    default:
      break
    }
    
    let w = pipelineStrong.threadExecutionWidth
    let h = pipelineStrong.maxTotalThreadsPerThreadgroup / w
    let threadsPerThreadgroup = MTLSizeMake(w, h, 1)
    
    let threadgroupsPerGrid = MTLSize(width: data.count / 2,
                                      height: data.count / 2,
                                      depth: 1)
    
    encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    //execution step
    cmds?.commit()
    cmds?.waitUntilCompleted()
    
    let dataArray = UnsafeMutableBufferPointer<Float>(start: resultsBuffer.contents().assumingMemoryBound(to: Float.self),
                                                      count: data.count)
    
    return Array(dataArray)
  }
  
}

