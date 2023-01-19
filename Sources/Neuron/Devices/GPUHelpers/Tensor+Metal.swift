//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/17/23.
//

import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate
import NumSwift

extension Tensor {
  static func asTexture(device: MTLDevice, commandQueue: MTLCommandQueue, size: TensorSize, usage: MTLTextureUsage) -> MTLTexture? {
    guard  let commandBuffer = commandQueue.makeCommandBuffer(),
           let encoder = commandBuffer.makeBlitCommandEncoder() else { return nil }
    
    let width = size.columns
    let height = size.rows
    let depth = size.depth
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float,
                                                              width: width,
                                                              height: height,
                                                              mipmapped: false)
    descriptor.arrayLength = depth
    descriptor.textureType = .type2DArray
    descriptor.usage = usage
    
    guard let texture = device.makeTexture(descriptor: descriptor) else { return nil }
    let bytesPerRow = MemoryLayout<Float>.stride * width
    let region = MTLRegionMake2D(0, 0, width, height)
    let bufferSize = bytesPerRow * height * depth
    
    var zeros = NumSwift.zerosLike((rows: size.rows, columns: size.columns, depth: size.depth))
    
    guard let buffer = device.makeBuffer(bytes: &zeros, length: bufferSize, options: []) else { return nil }
    
    for i in 0..<depth {
      encoder.copy(from: buffer,
                   sourceOffset: bytesPerRow * height * i,
                   sourceBytesPerRow: bytesPerRow,
                   sourceBytesPerImage: bytesPerRow * height,
                   sourceSize: region.size,
                   to: texture,
                   destinationSlice: i,
                   destinationLevel: 0,
                   destinationOrigin: region.origin)
    }
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return texture
  }
  
  func asTexture(device: MTLDevice, commandQueue: MTLCommandQueue, size: TensorSize, usage: MTLTextureUsage) -> MTLTexture? {
    guard  let commandBuffer = commandQueue.makeCommandBuffer(),
           let encoder = commandBuffer.makeBlitCommandEncoder() else { return nil }
    
    let width = size.columns
    let height = size.rows
    let depth = size.depth
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float,
                                                              width: width,
                                                              height: height,
                                                              mipmapped: false)
    descriptor.arrayLength = depth
    descriptor.textureType = .type2DArray
    descriptor.usage = usage
    
    guard let texture = device.makeTexture(descriptor: descriptor) else { return nil }
    let bytesPerRow = MemoryLayout<Float>.stride * width
    let region = MTLRegionMake2D(0, 0, width, height)
    let bufferSize = bytesPerRow * height * depth

    var zeros = value.flatten()
    guard let buffer = device.makeBuffer(bytes: &zeros, length: bufferSize, options: []) else { return nil }
    
    for i in 0..<depth {
      encoder.copy(from: buffer,
                   sourceOffset: bytesPerRow * height * i,
                   sourceBytesPerRow: bytesPerRow,
                   sourceBytesPerImage: bytesPerRow * height,
                   sourceSize: region.size,
                   to: texture,
                   destinationSlice: i,
                   destinationLevel: 0,
                   destinationOrigin: region.origin)
    }
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return texture
  }

}
