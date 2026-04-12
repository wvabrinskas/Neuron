//
//  File.swift
//
//
//  Created by William Vabrinskas on 5/4/22.
//

import Foundation
import NumSwift

/// A neural network activation layer that applies the Mish activation function.
///
/// Mish is a self-regularized non-monotonic activation defined as `f(x) = x * tanh(softplus(x))`.
/// It tends to outperform ReLU and Swish in deep networks by allowing small negative values to pass through.
public final class Mish: BaseActivationLayer {
  /// Creates a Mish activation layer.
  /// - Parameters:
  ///   - inputSize: The expected input tensor size. Defaults to an empty `TensorSize`.
  ///   - initializer: The weight initializer type. Defaults to `.heNormal`.
  ///   - linkId: A unique identifier for this layer. Defaults to a new UUID string.
  public init(inputSize: TensorSize = TensorSize(array: []),
              initializer: InitializerType = .heNormal,
              linkId: String = UUID().uuidString) {
    
    super.init(inputSize: inputSize,
               type: .mish,
               linkId: linkId,
               encodingType: .mish)
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize, type, linkId
  }
  
  /// Decodes a Mish activation layer from a serialized model.
  ///
  /// - Parameter decoder: Decoder used during model loading.
  /// - Throws: An error if required values cannot be decoded.
  convenience public required init(from decoder: Decoder) throws {
    self.init()
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.linkId = try container.decodeIfPresent(String.self, forKey: .linkId) ?? UUID().uuidString
    
    self.outputSize = inputSize
  }
  
  /// Encodes the layer's configuration into the given encoder.
  /// - Parameter encoder: The encoder to write layer data into.
  /// - Throws: An error if any value fails to encode.
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
    try container.encode(linkId, forKey: .linkId)
  }
  
  /// Performs the forward pass of the Mish activation: `f(x) = x * tanh(ln(1 + eˣ))`.
  /// - Parameters:
  ///   - tensor: The input tensor to activate.
  ///   - context: The network context for this pass (training vs inference).
  /// - Returns: A new tensor with the Mish activation applied element-wise.
  public override func forward(tensor: Tensor, context: NetworkContext) -> Tensor {
    let forward = tensor.storage
    let newStorage = TensorStorage.create(count: forward.count)
    let tanhForward = TensorStorage.create(count: forward.count)
    
    for i in 0..<newStorage.count {
      let value = forward[i]
      let tanhCalc = Tensor.Scalar.tanh(Tensor.Scalar.log(1 + Tensor.Scalar.exp(value)))
      tanhForward[i] = tanhCalc
      newStorage[i] = value * tanhCalc
    }
    
    let tensorContext = TensorContext { inputs, gradient, wrt in
      // backpropogation calculation
      
      let tanhSp = Tensor(storage: tanhForward, size: self.inputSize)
      
      let sigmoid = Sigmoid(inputSize: self.inputSize)
      let sigmoidOut = sigmoid(inputs).detached()
      
      let sech2 = (1 - tanhSp * tanhSp)
      
      let wrtInput = (tanhSp + inputs * sigmoidOut * sech2) * gradient
      wrtInput.label = "mish_input_gradient"
    
      return (wrtInput, Tensor(), Tensor())
    }
    
    // forward calculation - setGraph connects `tensor` so the custom context fires during backprop
    let out = Tensor(storage: newStorage, size: outputSize, context: tensorContext)
    out.label = "mish"
    out.setGraph(tensor)
    
    return out
  }
}

