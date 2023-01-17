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


/// A Tensor object who's value is set and retrieved asynchronously
/// When retrieveing the `.value` of this class it is possible that the value is not set at all as it is set asynchronously.
/// To be sure that the value is set use `getAsyncData()` async function. This will be guaranteed to return the value.
public final class AsyncTensor: Tensor {
  public enum AsyncTensorError: Error, LocalizedError {
    case timeoutReached
    
    public var errorDescription: String? {
      switch self {
      case .timeoutReached:
        return NSLocalizedString("Timeout was reached when waiting for Tensor to be set.", comment: "")
      }
    }
  }
  
  @Published
  public var asyncValue: Data?
  private var internalValue: Data = []
  private var cancellables: Set<AnyCancellable> = []
  
  override public var value: Tensor.Data {
    get {
      return internalValue
    }
    set {
      dataQueue.async(flags: .barrier) { [weak self] in
        self?.asyncValue = newValue
        self?.internalValue = newValue
      }
    }
  }
  
  private lazy var dataQueue = DispatchQueue(label: "async_tensor_\(id)",
                                             qos: .default,
                                             attributes: .concurrent,
                                             autoreleaseFrequency: .inherit,
                                             target: nil)
  
  /// Waits for the value of this Tensor to be set before returning. It has a timeout of 30 seconds.
  public func getDataAsync() async throws -> Data {
    try await withCheckedThrowingContinuation { continuation in
      dataQueue.async(flags: .barrier) { [weak self] in
        guard let self = self else {
          continuation.resume(returning: [])
          return
        }
        
        self.$asyncValue
          .removeDuplicates()
          .compactMap({ $0 })
          .setFailureType(to: AsyncTensorError.self)
          .timeout(30, scheduler: self.dataQueue, customError: { .timeoutReached })
          .sink(receiveCompletion: { completion in
            switch completion {
            case .failure(let error):
              continuation.resume(throwing: error)
            case .finished:
              break
            }
          }, receiveValue: { data in
            self.cancellables = []
            continuation.resume(returning: data)
          })
          .store(in: &self.cancellables)
      }
    }
  }
}
