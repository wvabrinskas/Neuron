//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/16/22.
//

import Foundation
import Metal
import NumSwift

/// A device abstraction representing GPU-accelerated computation.
///
/// Wraps the shared `GPUManager` and conforms to the `Device` protocol,
/// providing GPU-backed implementations of neural network operations.
public struct GPU: Device {
  /// Priority to run the multithreaded operations on
  public var qosPriority: DispatchQoS.QoSClass = .default
  /// The device type identifier for this implementation, set to `.cpu`.
  public var type: DeviceType = .gpu

  /// Creates a CPU-backed device implementation.
  public init() {}

  /// Calculates the convolution of the given inputs
  /// - Parameters:
  ///   - signal: The signal as a Tensor.Value array to perform the convolution on
  ///   - filter: The filter as a Tensor.Value array  to apply to the signal
  ///   - strides: Convolutional strides
  ///   - padding: Convolutional padding
  ///   - filterSize: Size of the filter
  ///   - inputSize: Size of the signal
  ///   - outputSize: Optional declaration of the output size of the convolution
  /// - Returns: The convolution result as a Tensor.Value array
  public func conv2d(signal: Tensor.Value,
                     filter: Tensor.Value,
                     strides: (Int, Int),
                     padding: NumSwift.ConvPadding,
                     filterSize: (rows: Int, columns: Int),
                     inputSize: (rows: Int, columns: Int),
                     outputSize: (rows: Int, columns: Int)?) -> Tensor.Value {
    NumSwiftFlat.conv2d(signal: signal,
                        filter: filter,
                        strides: strides,
                        padding: padding,
                        filterSize: filterSize,
                        inputSize: inputSize)
  }
  
  /// Calculates the transposed convolution of the given inputs
  /// - Parameters:
  ///   - signal: The signal as a Tensor.Value to perform the transposed convolution on
  ///   - filter: The filter as a Tensor.Value  to apply to the signal
  ///   - strides: Convolutional strides
  ///   - padding: Convolutional padding
  ///   - filterSize: Size of the filter
  ///   - inputSize: Size of the signal
  ///   - outputSize: Optional declaration of the output size of the transposed convolution
  /// - Returns: The transposed convolution result as a Tensor.Value
  public func transConv2d(signal: Tensor.Value,
                          filter: Tensor.Value,
                          strides: (Int, Int),
                          padding: NumSwift.ConvPadding,
                          filterSize: (rows: Int, columns: Int),
                          inputSize: (rows: Int, columns: Int),
                          outputSize: (rows: Int, columns: Int)?) -> Tensor.Value {
    NumSwiftFlat.transConv2d(signal: signal,
                             filter: filter,
                             strides: strides,
                             padding: padding,
                             filterSize: filterSize,
                             inputSize: inputSize)
  }
  
  /// Performs an activation function
  /// - Parameters:
  ///   - input: The input Tensor to perform the activation function on
  ///   - type: The type of activation to apply
  /// - Returns: The activated result as a Tensor
  public func activate(_ input: Tensor, _ type: Activation) -> Tensor {
    activate(input, type, encoder: nil)
  }

  public func activate(_ input: Tensor, _ type: Activation, encoder: MetalCommandEncoder?) -> Tensor {
    if let metalInput = input.storage as? MetalTensorStorage,
       metalInput.count >= Constants.metalElementwiseThreshold,
       MetalContext.shared.isAvailable,
       let device = MetalContext.shared.device,
       let pool = MetalContext.shared.bufferPool {
      let engine = MetalEngine()
      let outputStorage = MetalTensorStorage(device: device, count: metalInput.count, pool: pool)
      if let enc = encoder {
        if engine.encodeActivation(
          encoder: enc,
          input: metalInput,
          output: outputStorage,
          activationType: UInt32(type.index()),
          leakyAlpha: type.leakyAlphaForMetal
        ) {
          return Tensor(storage: outputStorage, size: input.size, context: TensorContext())
        }
      } else if engine.dispatchActivation(
        input: metalInput,
        output: outputStorage,
        activationType: UInt32(type.index()),
        leakyAlpha: type.leakyAlphaForMetal
      ) {
        return Tensor(storage: outputStorage, size: input.size, context: TensorContext())
      }
    }
    return type.activate(input: input)
  }

  /// Performs an activation derivative function
  /// - Parameters:
  ///   - input: The input Tensor to perform the activation derivative function on
  ///   - type: The type of activation derivative to apply
  /// - Returns: The activated derivative result as a Tensor
  public func derivate(_ input: Tensor, _ type: Activation) -> Tensor {
    derivate(input, type, encoder: nil)
  }

  public func derivate(_ input: Tensor, _ type: Activation, encoder: MetalCommandEncoder?) -> Tensor {
    if let metalInput = input.storage as? MetalTensorStorage,
       metalInput.count >= Constants.metalElementwiseThreshold,
       MetalContext.shared.isAvailable,
       let device = MetalContext.shared.device,
       let pool = MetalContext.shared.bufferPool {
      let engine = MetalEngine()
      let outputStorage = MetalTensorStorage(device: device, count: metalInput.count, pool: pool)
      if let enc = encoder {
        if engine.encodeDerivate(
          encoder: enc,
          input: metalInput,
          output: outputStorage,
          activationType: UInt32(type.index()),
          leakyAlpha: type.leakyAlphaForMetal
        ) {
          return Tensor(storage: outputStorage, size: input.size, context: TensorContext())
        }
      } else if engine.dispatchDerivate(
        input: metalInput,
        output: outputStorage,
        activationType: UInt32(type.index()),
        leakyAlpha: type.leakyAlphaForMetal
      ) {
        return Tensor(storage: outputStorage, size: input.size, context: TensorContext())
      }
    }
    return type.derivate(input)
  }

  /// Performs matrix multiplication for two tensors.
  ///
  /// - Parameters:
  ///   - a: Left tensor.
  ///   - b: Right tensor.
  /// - Returns: Matrix product tensor.
  public func matmul(_ a: Tensor, _ b: Tensor) -> Tensor {
    matmul(a, b, encoder: nil)
  }

  public func matmul(_ a: Tensor, _ b: Tensor, encoder: MetalCommandEncoder?) -> Tensor {
    if let metalA = a.storage as? MetalTensorStorage,
       let metalB = b.storage as? MetalTensorStorage,
       MetalContext.shared.isAvailable,
       let device = MetalContext.shared.device,
       let pool = MetalContext.shared.bufferPool {
      let aSize = a.size
      let bSize = b.size
      precondition(aSize.columns == bSize.rows, "A columns (\(aSize.columns)) must equal B rows (\(bSize.rows))")
      precondition(aSize.depth == bSize.depth, "A depth (\(aSize.depth)) must equal B depth (\(bSize.depth))")
      let M = aSize.rows
      let K = aSize.columns
      let N = bSize.columns
      let depth = aSize.depth
      let mkn = M * K * N * depth
      guard mkn >= Constants.metalMatmulThreshold else {
        return a.matmul(b)
      }
      let cCount = M * N * depth
      let engine = MetalEngine()
      let outputStorage = MetalTensorStorage(device: device, count: cCount, pool: pool)
      if let enc = encoder {
        if engine.encodeMatmul(encoder: enc, a: metalA, b: metalB, output: outputStorage, M: M, N: N, K: K, depth: depth) {
          return Tensor(storage: outputStorage, size: TensorSize(rows: M, columns: N, depth: depth), context: TensorContext())
        }
      } else if engine.dispatchMatmul(a: metalA, b: metalB, output: outputStorage, M: M, N: N, K: K, depth: depth) {
        return Tensor(storage: outputStorage, size: TensorSize(rows: M, columns: N, depth: depth), context: TensorContext())
      }
    }
    return a.matmul(b)
  }
}
