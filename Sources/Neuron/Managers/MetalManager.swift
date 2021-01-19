import Foundation
import Metal
import MetalPerformanceShaders

public typealias DataType = [[CFloat]]
public typealias ResultType = CFloat
public typealias InputType = Float32

public class MetalManager {
  public static let shared = MetalManager()
  
  public enum MetalFunction: String {
    case parsum
    case activation
  }
  
  private var currentRunningPipelines: [MTLComputePipelineState] = []
  private var device: MTLDevice? = MTLCreateSystemDefaultDevice()
  
  private let maxNumberOfQueues: Int = 1
  
  private var queues: [MTLCommandQueue] = []
  
  private func getQueue() -> MTLCommandQueue? {
    if queues.count == maxNumberOfQueues {
      return self.queues.randomElement()
    }
    
    guard let queue = device?.makeCommandQueue() else {
      return nil
    }
    
    self.queues.append(queue)
    return queue
  }

  private func getFunction(_ function: MetalFunction) -> MTLFunction? {
    return device?.makeDefaultLibrary()?.makeFunction(name: function.rawValue)
  }
 
  
  public func massiveMatrix(nodeCount: Int, inputValues: [InputType], weights: [InputType]) -> [Float] {
    guard let device = self.device else {
      print("Could not create metal device")
      return []
    }
    
    let queue = self.getQueue()
    let cmds = queue?.makeCommandBuffer()

    let m = MPSMatrixFullyConnected(device: device)

    let leftLength = MemoryLayout<InputType>.stride * inputValues.count
    
    guard let buffer = device.makeBuffer(bytes: inputValues,
                                         length: leftLength,
                                         options: .storageModeShared) else {
      return []
    }
    
    let leftDescriptor = MPSMatrixDescriptor(rows: nodeCount,
                                             columns: inputValues.count / nodeCount,
                                             rowBytes: leftLength / nodeCount,
                                             dataType: .float32)
    
    let leftMatrix = MPSMatrix(buffer: buffer, descriptor: leftDescriptor)
    
    let rightLength = MemoryLayout<InputType>.stride * weights.count
    
    guard let rightBuffer = device.makeBuffer(bytes: weights,
                                              length: rightLength,
                                              options: .storageModeShared) else {
      return []
    }
    
    let rightDescriptor = MPSMatrixDescriptor(rows: weights.count / nodeCount,
                                              columns: nodeCount,
                                              rowBytes: rightLength / weights.count * nodeCount,
                                              dataType: .float32)
    
    let rightMatrix = MPSMatrix(buffer: rightBuffer, descriptor: rightDescriptor)
    
    let resultLength = MemoryLayout<InputType>.stride * leftMatrix.rows * rightMatrix.columns
    
    guard let resultBuffer = device.makeBuffer(length: resultLength, options: .storageModeShared) else {
      return []
    }
    
    let resultMatrixDescriptor = MPSMatrixDescriptor(rows: leftMatrix.rows,
                                                     columns: rightMatrix.columns,
                                                     rowBytes: resultLength / leftMatrix.rows,
                                                     dataType: .float32)
    
    let resultMatrix = MPSMatrix(buffer: resultBuffer, descriptor: resultMatrixDescriptor)
    
    guard let command = cmds else {
      assertionFailure("no command queue available...")
      return []
    }
    
    m.encode(commandBuffer: command,
             inputMatrix: leftMatrix,
             weightMatrix: rightMatrix,
             biasVector: nil,
             resultMatrix: resultMatrix)
    
    command.commit()
    command.waitUntilCompleted()
    
    let rawPointer = resultMatrix.data.contents()
    let typePointer = rawPointer.bindMemory(to: InputType.self, capacity: resultMatrix.columns * resultMatrix.rows)
    let bufferPointer = UnsafeBufferPointer(start: typePointer, count: resultMatrix.columns * resultMatrix.rows)
    
    var outputs: [InputType] = []
    let _ = bufferPointer.map({ outputs += [$0] })

    return outputs
  }
}
