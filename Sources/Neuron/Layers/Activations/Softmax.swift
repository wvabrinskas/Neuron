//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/30/22.
//

import Foundation
import NumSwift
import Numerics

/// Softmax activation layer for multi-class classification
/// Applies the softmax function: f(x_i) = e^(x_i) / Σ(e^(x_j)) for all j
/// Converts a vector of real values into a probability distribution
/// Output values sum to 1, making it ideal for classification tasks
public final class Softmax: BaseActivationLayer {
  /// Coding keys for serialization
  enum CodingKeys: String, CodingKey {
    case inputSize,
         type
  }
  
  /// Initializes Softmax layer from decoder for deserialization
  /// - Parameter decoder: Decoder containing serialized layer data
  /// - Throws: Decoding errors if deserialization fails
  convenience required public init(from decoder: Decoder) throws {
    self.init()
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.outputSize = inputSize
  }
  
  /// Encodes the Softmax layer for serialization
  /// - Parameter encoder: Encoder to serialize layer data
  /// - Throws: Encoding errors if serialization fails
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
  }
  
  /// Initializes a Softmax activation layer
  /// Commonly used as the final layer in multi-class classification networks
  /// - Parameter inputSize: Optional input tensor size. Required for first layer in network
  public init(inputSize: TensorSize = TensorSize(array: [])) {
    super.init(inputSize: inputSize,
               type: .softmax,
               encodingType: .softmax)
  }
  
  /// Performs forward pass through the softmax layer
  /// Computes softmax probabilities with numerical stability (subtracting max value)
  /// - Parameters:
  ///   - tensor: Input tensor to apply softmax to
  ///   - context: Network context for computation
  /// - Returns: Output tensor with softmax probabilities (values sum to 1)
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    let context = TensorContext { inputs, gradient in
      return (Tensor(gradient.value), Tensor(), Tensor())
    }
    
    var activationResult: [[[Tensor.Scalar]]] = []
    
    tensor.value.forEach { d in
      var row: [[Tensor.Scalar]] = []
      d.forEach { r in
        var column: [Tensor.Scalar] = []
        for i in 0..<r.count {
          column.append(calculate(index: i, outputs: r))
        }
        row.append(column)
      }
      activationResult.append(row)
    }
    
    let out = Tensor(activationResult, context: context)
    out.label = type.asString()
    
    out.setGraph(tensor)

    return out
  }
  
  /// Calculates softmax probability for a specific index
  /// Uses numerical stability technique by subtracting the maximum value
  /// - Parameters:
  ///   - index: Index of the element to compute softmax for
  ///   - outputs: Array of all logit values
  /// - Returns: Softmax probability for the specified index
  private func calculate(index: Int, outputs: [Tensor.Scalar]) -> Tensor.Scalar {
    let max = outputs.max() ?? 1
    var sum: Tensor.Scalar = 0
    outputs.forEach { (output) in
      sum += Tensor.Scalar.pow(Tensor.Scalar(Darwin.M_E), output - max)
    }
    
    return Tensor.Scalar.pow(Tensor.Scalar(Darwin.M_E), outputs[index] - max) / sum
  }
  
  /// Called when input size is set, configures output size
  /// For Softmax, output size equals input size (preserves dimensions)
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = inputSize
  }
}

