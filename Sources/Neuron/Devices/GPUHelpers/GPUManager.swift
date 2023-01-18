import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate
import NumSwift

public typealias DataType = [[CFloat]]
public typealias ResultType = CFloat

public class GPUManager {
  public enum MetalFunction: String {
    case activation, derivation, conv2d
  }
  
  private var currentRunningPipelines: [String: MTLComputePipelineState] = [:]
  private var device: MTLDevice? = MTLCreateSystemDefaultDevice()

  lazy var queue = self.device?.makeCommandQueue(maxCommandBufferCount: 64)
  lazy var cmds = queue?.makeCommandBuffer()
  
  private func getFunction(_ function: MetalFunction) -> MTLFunction? {
    return try? device?.makeDefaultLibrary(bundle: Bundle.module).makeFunction(name: function.rawValue)
  }
  
  private func pipelineIfExists(type: MetalFunction) -> MTLComputePipelineState? {
    return self.currentRunningPipelines[type.rawValue]
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
      
      self.currentRunningPipelines[type.rawValue] = pipeline
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
                                inputSize: TensorSize) -> TensorSize {
    let paddingValue = padding.extra(inputSize: (inputSize.rows, inputSize.columns), filterSize: filterSize)
    
    let rows = (((inputSize.rows + (paddingValue.top + paddingValue.bottom)) - (filterSize.rows - 1) - 1) / strides.rows) + 1
    let columns = (((inputSize.columns + (paddingValue.left + paddingValue.right)) - (filterSize.columns - 1) - 1) / strides.columns) + 1
    
    return TensorSize(array: [columns, rows, filterCount])
  }
  
  func activate(_ input: Tensor,
                inputSize: TensorSize,
                activationType: Activation,
                derivate: Bool = false) -> Tensor {
    
    autoreleasepool {
      
      guard let device = self.device, let queue = queue else {
        return Tensor()
      }
      
      let inputTexture = input.asTexture(device: device,
                                         commandQueue: queue,
                                         size: inputSize,
                                         usage: .shaderRead)
      
      let outputTexture = Tensor().asTexture(device: device,
                                             commandQueue: queue,
                                             size: inputSize,
                                             usage: .shaderWrite)
      
      let function: MetalFunction = derivate ? .derivation : .activation
      let pipeline: MTLComputePipelineState? = self.pipelineIfExists(type: function) ?? self.addPipeline(for: function)
      
      let newEncoder = cmds?.makeComputeCommandEncoder()
      
      guard let encoder = newEncoder, let pipelineStrong = pipeline else {
        return Tensor()
      }
      
      var activation = CUnsignedInt(activationType.index())
      
      encoder.setComputePipelineState(pipelineStrong)
      encoder.setTexture(inputTexture, index: 0)
      encoder.setTexture(outputTexture, index: 1)
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
      
      let threadgroupsPerGrid = MTLSize(width: max(1, inputSize.columns / 2),
                                        height: max(1, inputSize.rows / 2),
                                        depth: inputSize.depth)
      
      encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
      encoder.endEncoding()
      
      //execution step
      cmds?.commit()
      cmds?.waitUntilCompleted()
      
      let tensor = outputTexture?.get3d(commandQueue: queue, device: device)
      let out = Tensor(tensor ?? [])
      return out
    }
  }
  
  public func conv2d(_ input: Tensor,
                     filters: [Tensor],
                     biases: [Tensor.Scalar] = [],
                     padding: NumSwift.ConvPadding,
                     filterSize: (rows: Int, columns: Int),
                     strides: (rows: Int, columns: Int),
                     inputSize: TensorSize) -> Tensor {
    
    guard let queue = self.queue, let device = self.device else { return Tensor() }
    
    let outputSize = conv2dOutputSize(padding: padding,
                                      strides: strides,
                                      filterCount: filters.count,
                                      filterSize: filterSize,
                                      inputSize: inputSize)
    
    var textures: [MTLTexture] = []
    filters.forEach { filter in
      if let texture = conv2d(input,
                           filter: filter,
                           padding: padding,
                           filterSize: filterSize,
                           strides: strides,
                              inputSize: inputSize) {
        textures.append(texture)
      }
    }
    
    cmds?.commit()
    cmds?.waitUntilCompleted()
    
    var i = 0
    let tensor = Tensor(textures.map { texture in
      let out = texture.get3d(commandQueue: queue, device: device)
      let reducableStartingArray = [[Float]].init(repeating: [Float].init(repeating: 0,
                                                                          count: outputSize.columns),
                                                  count: outputSize.rows)
      let outSummed = out.reduce(reducableStartingArray, +)
      let bias = biases[safe: i, 0]
      i += 1
      return outSummed + bias
    })
    
    return tensor
  }
  
  /// returns a 3D tensor where each element is conv on the input with a filter with a depth of 1
  public func conv2d(_ input: Tensor,
                     filter: Tensor,
                     padding: NumSwift.ConvPadding,
                     filterSize: (rows: Int, columns: Int),
                     strides: (rows: Int, columns: Int),
                     inputSize: TensorSize) -> MTLTexture? {
    
    autoreleasepool {
      
      guard let device = self.device, let queue = queue else {
        return nil
      }
      
      let function: MetalFunction = .conv2d
      let pipeline: MTLComputePipelineState? = self.pipelineIfExists(type: function) ?? self.addPipeline(for: function)
      
      let newEncoder = cmds?.makeComputeCommandEncoder()
      
      guard let encoder = newEncoder, let pipelineStrong = pipeline else {
        return nil
      }
      
      let outputSize = conv2dOutputSize(padding: padding,
                                        strides: strides,
                                        filterCount: 1,
                                        filterSize: filterSize,
                                        inputSize: inputSize)
      
      let inputTexture = input.asTexture(device: device,
                                         commandQueue: queue,
                                         size: inputSize,
                                         usage: .shaderRead)
      
      let outputTexture = Tensor().asTexture(device: device,
                                             commandQueue: queue,
                                             size: TensorSize(rows: outputSize.rows,
                                                              columns: outputSize.columns,
                                                              depth: inputSize.depth), //use inputSize.depth here because we can parallelize the convolutions but not the summation
                                             usage: .shaderWrite)
      
      let filterTexture = filter.asTexture(device: device,
                                           commandQueue: queue,
                                           size: TensorSize(rows: filterSize.rows,
                                                            columns: filterSize.columns,
                                                            depth: inputSize.depth),
                                           usage: .shaderRead)
      
      encoder.setComputePipelineState(pipelineStrong)
      encoder.setTexture(inputTexture, index: 0)
      encoder.setTexture(outputTexture, index: 1)
      encoder.setTexture(filterTexture, index: 2)
      
      var inSize = SIMD3(CUnsignedInt(inputSize.rows), CUnsignedInt(inputSize.columns), CUnsignedInt(inputSize.depth))
      var outSize = SIMD3(CUnsignedInt(outputSize.rows), CUnsignedInt(outputSize.columns), CUnsignedInt(outputSize.depth))
      var kSize = SIMD2(CUnsignedInt(filterSize.rows), CUnsignedInt(filterSize.columns))
      var strides = SIMD2(CUnsignedInt(strides.rows), CUnsignedInt(strides.columns))
      var padding = padding == .same ? 1 : 0
      
      encoder.setBytes(&inSize, length: MemoryLayout<SIMD3<CUnsignedInt>>.size, index: 1)
      encoder.setBytes(&outSize, length: MemoryLayout<SIMD3<CUnsignedInt>>.size, index: 2)
      encoder.setBytes(&kSize, length: MemoryLayout<SIMD2<CUnsignedInt>>.size, index: 3)
      encoder.setBytes(&strides, length: MemoryLayout<SIMD2<CUnsignedInt>>.size, index: 4)
      encoder.setBytes(&padding, length: MemoryLayout<Int>.size, index: 5)
      
      let w = pipelineStrong.threadExecutionWidth
      let h = pipelineStrong.maxTotalThreadsPerThreadgroup / w
      let threadsPerThreadgroup = MTLSizeMake(w, h, 1)
      
      let threadgroupsPerGrid = MTLSize(width: max(1, inputSize.columns / 2),
                                        height: max(1, inputSize.rows / 2),
                                        depth: inputSize.depth)
      
      encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
      encoder.endEncoding()
  
      return outputTexture
    }
  }
  
}


