//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/15/23.
//

import Foundation
import NumSwift
import NumSwiftC
import Combine


public final class AsyncTensor: Tensor {
  private lazy var dataQueue = DispatchQueue(label: "async_tensor_\(id)",
                                             qos: .default,
                                             attributes: .concurrent,
                                             autoreleaseFrequency: .inherit,
                                             target: nil)
  public func setDataAsync(_ data: Data) {
    dataQueue.async(flags: .barrier) { [weak self] in
      self?.value = data
    }
  }
  
  public func getDataAsync() async -> Data {
    await withCheckedContinuation { continuation in
      dataQueue.async(flags: .barrier) { [weak self] in
        continuation.resume(returning: self?.value ?? [])
      }
    }
  }
}
