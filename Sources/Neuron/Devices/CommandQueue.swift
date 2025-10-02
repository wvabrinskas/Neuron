//
//  CommandQueue.swift
//  Neuron
//
//  Created by William Vabrinskas on 8/21/25.
//

import Foundation

public struct Command {
  let action: () -> [[Tensor.Scalar]]
}

public final class CommandQueue {
  public var onExecuteQueue: () -> ()
  public var processingCount: Int

  private let maxIterations = Constants.maxWorkers
  private var executionQueue: Set<UUID> = []

  private let lock = NSLock()
  private let group = DispatchGroup()
  
  public init(_ onExecuteQueue: (() -> Void)? = nil, processingCount: Int = 1) {
    if let onExecuteQueue = onExecuteQueue {
      self.onExecuteQueue = onExecuteQueue
    } else {
      self.onExecuteQueue = { }
    }
    
    self.processingCount = processingCount
  }
  
  public func enqueue(_ command: @escaping (_ result: @escaping ([[Tensor.Scalar]]) -> ()) -> ()) -> [[Tensor.Scalar]] {
    let uuid = UUID()
        
    var result: [[Tensor.Scalar]] = []
    
    group.enter()

    command({ [weak self] completion in
      guard let self else { return }
      
      result = completion
      group.leave()
    })
    
    // we now commit the second we call the command
    // the issue is once we hold up this thread none of the other inputs
    // will be processed because this thread is being held up.
    // we need to submit all of the operations and then wait on them somehow
    
    group.wait()
    

    return result
  }
  
}
