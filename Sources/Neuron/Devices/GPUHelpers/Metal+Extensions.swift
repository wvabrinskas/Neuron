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

extension MTLTexture {
  func get3d(commandQueue: MTLCommandQueue, device: MTLDevice) -> [[[Float]]] {
    guard  let commandBuffer = commandQueue.makeCommandBuffer(),
           let encoder = commandBuffer.makeBlitCommandEncoder() else { return [] }
    
    let width = width
    let height = height
    let depth = arrayLength
    let bytesPerRow = MemoryLayout<Float>.stride * width
    let region = MTLRegionMake2D(0, 0, width, height)
    var array3D = NumSwift.zerosLike((width, height, depth))
    let bufferSize = bytesPerRow * height * depth
    guard let buffer = device.makeBuffer(length: bufferSize, options: []) else { return array3D }
    
    encoder.synchronize(resource: self)
    for i in 0..<depth {
      encoder.copy(from: self,
                   sourceSlice: i,
                   sourceLevel: 0,
                   sourceOrigin: region.origin,
                   sourceSize: region.size,
                   to: buffer,
                   destinationOffset: bytesPerRow * height * i,
                   destinationBytesPerRow: bytesPerRow,
                   destinationBytesPerImage: bytesPerRow * height)
    }
    
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let bufferPointer = buffer.contents().bindMemory(to: Float.self, capacity: bufferSize/MemoryLayout<Float>.stride)
    for i in 0..<depth {
      for j in 0..<height {
        for k in 0..<width {
          array3D[i][j][k] = bufferPointer[i*height*width + j*width + k]
        }
      }
    }
    return array3D
  }
}
