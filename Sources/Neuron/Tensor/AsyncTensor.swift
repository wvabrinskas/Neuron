//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/15/23.
//

import Foundation
import Combine


/// A Tensor object who's value is set and retrieved asynchronously
/// When retrieveing the `.value` of this class it is possible that the value is not set at all as it is set asynchronously.
/// To be sure that the value is set use `getAsyncData()` async function. This will be guaranteed to return the value.
public final class AsyncTensor: Tensor {
  public enum AsyncTensorError: Error, LocalizedError {
    case timeoutReached
    case selfLost
    
    public var errorDescription: String? {
      switch self {
      case .timeoutReached:
        return NSLocalizedString("Timeout was reached when waiting for Tensor to be set.", comment: "")
      case .selfLost:
        return NSLocalizedString("This tensor object has been deallocated before its value has been set.", comment: "")
      }
    }
  }
  
  @Published
  public var asyncValue: Data?
  public var timeout: DispatchQueue.SchedulerTimeType.Stride = .seconds(30)
  private var internalValue: Data = []
  private var cancellables: Set<AnyCancellable> = []
  private let semaphore: DispatchSemaphore = .init(value: 0)
  
  override public var value: Tensor.Data {
    get {
      return internalValue
    }
    set {
      dataQueue.async(flags: .barrier) { [weak self] in
        self?.semaphore.signal()
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
  
  /// Waits for the value of this Tensor to be set before returning. Timeout is default to 30 seconds
  /// - Returns: The data attached to the Tensor or an error
  public func getDataAsync() async throws -> Data {
    try await withCheckedThrowingContinuation { continuation in
      dataQueue.async(flags: .barrier) { [weak self] in
        guard let self = self else {
          continuation.resume(throwing: AsyncTensorError.selfLost)
          return
        }
        
        self.$asyncValue
          .compactMap({ $0 })
          .setFailureType(to: AsyncTensorError.self)
          .timeout(self.timeout, scheduler: self.dataQueue, customError: { .timeoutReached })
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
  
  /// This function will block the current thread until the Tensor's value is set.
  /// - Parameter onError: will be called if there's an error while waiting for value being set. This could include the timeout being reached.
  public func wait(onError: ((AsyncTensorError?) -> ())? = nil) {
    
    dataQueue.asyncAfter(deadline: .now() + self.timeout.timeInterval) { [weak self] in
      self?.semaphore.signal()
      onError?(.timeoutReached)
    }
    
    semaphore.wait()
  }
}
