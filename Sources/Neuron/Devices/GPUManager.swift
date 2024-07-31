import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate


// MARK: WIP - This is a work in progress and is not ready for use.

extension MTLTexture {
  func getPixels<T> (_ region: MTLRegion? = nil, mipmapLevel: Int = 0) -> UnsafeMutablePointer<T> {
    let fromRegion  = region ?? MTLRegionMake2D(0, 0, self.width, self.height)
    let width       = fromRegion.size.width
    let height      = fromRegion.size.height
    let bytesPerRow = MemoryLayout<T>.stride * width
    let data        = UnsafeMutablePointer<T>.allocate(capacity: bytesPerRow * height)
    
    self.getBytes(data, bytesPerRow: bytesPerRow, from: fromRegion, mipmapLevel: mipmapLevel)
    return data
  }
}

public typealias DataType = [[CFloat]]
public typealias ResultType = CFloat

public class GPUManager {
  public static let shared = GPUManager()
  
  public enum MetalFunction: String {
    case activation, derivate
  }
  
  private var currentRunningPipelines: [MTLComputePipelineState] = []
  private var device: MTLDevice? = MTLCreateSystemDefaultDevice()
  
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
//  
//  public func conv2d(_ input: Tensor<N>,
//                     filters: [Tensor<N>],
//                     biases: Tensor<N>,
//                     filterCount: Int,
//                     filterSize: (rows: Int, columns: Int),
//                     strides: (rows: Int, columns: Int),
//                     inputSize: (rows: Int, columns: Int, depth: Int),
//                     outputSize: (rows: Int, columns: Int, depth: Int)) {
//    
//    let descriptor = MPSCNNConvolutionDescriptor(kernelWidth: filterSize.columns,
//                                                 kernelHeight: filterSize.rows,
//                                                 inputFeatureChannels: inputSize.depth,
//                                                 outputFeatureChannels: filterCount)
//    
//    
//    descriptor.strideInPixelsX = strides.columns
//    descriptor.strideInPixelsY = strides.rows
//    
//    guard let device = device else {
//      return
//    }
//    
//    let filtersFlat = filters.map { $0.value }.flatten().flatten()
//    let biasesFlat = biases.value.flatten()
//    
//    let conv = MPSCNNConvolution(device: device,
//                                 convolutionDescriptor: descriptor,
//                                 kernelWeights: filtersFlat,
//                                 biasTerms: biasesFlat,
//                                 flags: .none)
//
//    let inputImageDescriptor  = MPSImageDescriptor(channelFormat: MPSImageFeatureChannelFormat.float16,
//                                                   width: inputSize.columns,
//                                                   height: inputSize.rows,
//                                                   featureChannels: inputSize.depth)
//    
//    let image = MPSImage(device: device, imageDescriptor: inputImageDescriptor)
//    let inputImageData = input.value.flatten()
//    
//    image.texture.replace(region: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
//                                            size: MTLSize(width: inputSize.columns,
//                                                          height: inputSize.rows,
//                                                          depth: inputSize.depth)),
//                          mipmapLevel: 0,
//                          withBytes: inputImageData,
//                          bytesPerRow: inputSize.columns * MemoryLayout<Tensor<N>.Scalar>.stride)
//    
//    guard let queue = device.makeCommandQueue(),
//          let commandBuffer = queue.makeCommandBuffer() else {
//      return
//    }
//    
//    let outputImageDescriptor  = MPSImageDescriptor(channelFormat: MPSImageFeatureChannelFormat.float16,
//                                                    width: outputSize.columns,
//                                                    height: outputSize.rows,
//                                                    featureChannels: outputSize.depth)
//    
//    let outImage = MPSImage(device: commandBuffer.device, imageDescriptor: outputImageDescriptor)
//    
//    conv.encode(commandBuffer: commandBuffer, sourceImage: image, destinationImage: outImage)
//    
//    commandBuffer.commit()
//    commandBuffer.waitUntilCompleted()
//    
//    //let r = Array(arrayLiteral: outImage.texture.buffer?.contents())
//    
//    let pixels: UnsafeMutablePointer<Tensor<N>.Scalar> = image.texture.getPixels()
//    
//    defer {
//      pixels.deallocate()
//    }
//    
//    var result: [Tensor<N>.Scalar] = []
//    
//    let capacity = outputSize.columns * outputSize.rows * MemoryLayout<Tensor<N>.Scalar>.stride
//
//    for i in stride(from: 0, to: capacity, by: 4) {
//      let l     = pixels[i + 0]
//      let a     = pixels[i + 1]
//      let b     = pixels[i + 2]
//      let alpha = pixels[i + 3]
//      
//      print(l, a, b, alpha)
//      
//      result.append(a)
//    }
//    
//    print(result)
//  }
//  
  public func activate<N: TensorNumeric>(_ num: [Tensor<N>.Scalar],
                       _ activationType: Activation,
                       derivate: Bool = false) -> [Tensor<N>.Scalar] {
    var data = num
    
    guard let device = self.device else {
      return num
    }
    
    let function: MetalFunction = derivate ? .derivate : .activation
    
    let pipeline: MTLComputePipelineState? = self.pipelineIfExists(type: function) ?? self.addPipeline(for: function)
    
    guard let dataBuffer = device.makeBuffer(bytes: &data,
                                             length: MemoryLayout<Tensor<N>.Scalar>.stride * data.count,
                                             options: []),
          
            let resultsBuffer = device.makeBuffer(length: MemoryLayout<Tensor<N>.Scalar>.stride * data.count,
                                                  options: []) else {
      return num
    }
    
    let queue = self.device?.makeCommandQueue()
    let cmds = queue?.makeCommandBuffer()
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
      var limit = 0//Tensor<N>.Scalar(limit)
      encoder.setBytes(&limit, length: MemoryLayout<Tensor<N>.Scalar>.size, index: 3)
    default:
      break
    }
    
    let w = pipelineStrong.threadExecutionWidth
    let _ = pipelineStrong.maxTotalThreadsPerThreadgroup / w
    let threadsPerThreadgroup = MTLSizeMake(1, 1, 1)
    
    let threadgroupsPerGrid = MTLSize(width: 1,
                                      height: 1,
                                      depth: 1)
    
    encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    
    //execution step
    cmds?.commit()
    cmds?.waitUntilCompleted()
    
    let dataArray = UnsafeMutableBufferPointer<Tensor<N>.Scalar>(start: resultsBuffer.contents().assumingMemoryBound(to: Tensor<N>.Scalar.self),
                                                      count: data.count)
    
    return Array(dataArray)
  }
  
}

