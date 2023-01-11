import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate


extension MTLTexture {
  func getValues(device: MTLDevice, region: MTLRegion? = nil) -> [Float] {
    let region = region ?? MTLRegionMake2D(0, 0, self.width, self.height)
    let rowBytes = region.size.width * MemoryLayout<Float>.stride
    let byteCount = region.size.height * rowBytes
    let totalSize = region.size.width * region.size.height
    guard let buffer = device.makeBuffer(length: byteCount, options: []) else { return [] }
    getBytes(buffer.contents(), bytesPerRow: rowBytes, from: region, mipmapLevel: 0)
    let values = Array(UnsafeBufferPointer(start: buffer.contents().bindMemory(to: Float.self, capacity: byteCount), count: totalSize))
    return values
  }
}

public typealias DataType = [[CFloat]]
public typealias ResultType = CFloat

public class GPUManager {
  public static let shared = GPUManager()
  
  public enum MetalFunction: String {
    case activation, derivate, conv2d
  }
  
  private var currentRunningPipelines: [MTLComputePipelineState] = []
  private var device: MTLDevice? = MTLCreateSystemDefaultDevice()
  
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
  
  public func commit() {
    
  }
  
  public func conv2d(_ input: Tensor,
                     filters: [Tensor],
                     biases: Tensor,
                     filterCount: Int,
                     filterSize: (rows: Int, columns: Int),
                     strides: (rows: Int, columns: Int),
                     inputSize: (rows: Int, columns: Int, depth: Int),
                     outputSize: (rows: Int, columns: Int, depth: Int)) {
    
    guard let device = device else {
      return
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
    
    guard let inputTexture = device.makeTexture(descriptor: inputTextureDesc) else { return }
    
    let flatInput: [Float] = input.value.flatten()
    let pointer = UnsafeMutableRawPointer(mutating: flatInput)
    let bytesPerRow = inputTexture.width * MemoryLayout<Float>.stride
    let region = MTLRegionMake2D(0, 0, inputTexture.width, inputTexture.height)
    inputTexture.replace(region: region, mipmapLevel: 0, withBytes: pointer, bytesPerRow: bytesPerRow)
    
    let input = inputTexture.getValues(device: device, region: region)
    print(input)
    print(input.count)
    
    // output texture
    let outputTextureDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float,
                                                                    width: outputSize.columns,
                                                                    height: outputSize.rows,
                                                                    mipmapped: false)
    
    let outputTexture = device.makeTexture(descriptor: outputTextureDesc)
    
    var filtersFlat: [Float] = filters.map { $0.value}.fullFlatten()
   
    let newEncoder = cmds?.makeComputeCommandEncoder()
    
    guard let encoder = newEncoder, let pipelineStrong = pipeline else {
      return
    }
    
    guard let filterBuffer = device.makeBuffer(bytes: &filtersFlat,
                                               length: MemoryLayout<Float>.stride * filtersFlat.count,
                                               options: []) else {
      return
    }
        
    encoder.setComputePipelineState(pipelineStrong)
    encoder.setTexture(inputTexture, index: 0)
    encoder.setTexture(outputTexture, index: 1)
    
    encoder.setBuffer(filterBuffer, offset: 0, index: 0)
    
    var inSize = SIMD2(CUnsignedInt(inputSize.rows), CUnsignedInt(inputSize.columns))
    var outSize = SIMD2(CUnsignedInt(outputSize.rows), CUnsignedInt(outputSize.columns))
    var kSize = SIMD2(CUnsignedInt(filterSize.rows), CUnsignedInt(filterSize.columns))
    var strides = SIMD2(CUnsignedInt(strides.rows), CUnsignedInt(strides.columns))
    var padding = SIMD2(CUnsignedInt(0), CUnsignedInt(0))
    
    encoder.setBytes(&inSize, length: MemoryLayout<SIMD2<CUnsignedInt>>.size, index: 1)
    encoder.setBytes(&outSize, length: MemoryLayout<SIMD2<CUnsignedInt>>.size, index: 2)
    encoder.setBytes(&kSize, length: MemoryLayout<SIMD2<CUnsignedInt>>.size, index: 3)
    encoder.setBytes(&strides, length: MemoryLayout<SIMD2<CUnsignedInt>>.size, index: 4)
    encoder.setBytes(&padding, length: MemoryLayout<SIMD2<CUnsignedInt>>.size, index: 5)

    let w = pipelineStrong.threadExecutionWidth
    let h = pipelineStrong.maxTotalThreadsPerThreadgroup / w
    let threadsPerThreadgroup = MTLSizeMake(w, h, 1)
    
    let threadgroupsPerGrid = MTLSize(width: inputSize.columns,
                                      height: inputSize.rows,
                                      depth: inputSize.depth)
    
    encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()

    //execution step
    cmds?.commit()
    cmds?.waitUntilCompleted()
    
    let out = outputTexture?.getValues(device: device)
    print(out?.reshape(columns: outputSize.columns))
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

