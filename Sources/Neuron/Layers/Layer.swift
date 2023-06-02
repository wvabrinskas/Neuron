//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/26/22.
//

import Foundation
import NumSwift
import NumSwiftC

/// Layer types
public enum EncodingType: String, Codable {
  case leakyRelu,
       relu,
       sigmoid,
       softmax,
       swish,
       tanh,
       batchNormalize,
       conv2d,
       dense,
       dropout,
       flatten,
       maxPool,
       reshape,
       transConv2d,
       layerNormalize,
       lstm
}

/// A layer that performs an activation function
public protocol ActivationLayer: Layer {
  var type: Activation { get }
}

/// A layer that performs a convolution operation
public protocol ConvolutionalLayer: Layer {
  var filterCount: Int { get }
  var filters: [Tensor] { get }
  var filterSize: (rows: Int, columns: Int) { get }
  var strides: (rows: Int, columns: Int) { get }
  var padding: NumSwift.ConvPadding { get }
}

/// The the object that perform ML operations
public protocol Layer: AnyObject, Codable {
  var encodingType: EncodingType { get set }
  var extraEncodables: [String: Codable]? { get }
  var inputSize: TensorSize { get set }
  var outputSize: TensorSize { get }
  var weights: Tensor { get }
  var biases: Tensor { get }
  var biasEnabled: Bool { get set }
  var trainable: Bool { get set }
  var initializer: Initializer? { get }
  var device: Device { get set }
  func forward(tensor: Tensor) -> Tensor
  func apply(gradients: Optimizer.Gradient, learningRate: Float)
}

extension Layer {
  public var extraEncodables: [String: Codable]? {
    return [:]
  }

}
