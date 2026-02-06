//
//  File.swift
//
//
//  Created by William Vabrinskas on 4/30/22.
//

import Foundation
import NumSwift
import Numerics

/// Performs a Softmax activation.
public final class Softmax: BaseActivationLayer {
  enum CodingKeys: String, CodingKey {
    case inputSize,
         type
  }
  
  convenience required public init(from decoder: Decoder) throws {
    self.init()
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.outputSize = inputSize
  }
  
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
  }
  
  /// Default initializer for a Softmax activation.
  /// - Parameter inputSize: Optional input size at this layer. If this is the first layer you will need to set this.
  public init(inputSize: TensorSize? = nil) {
    super.init(inputSize: inputSize,
               type: .softmax,
               encodingType: .softmax)
  }
  
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    let context = TensorContext { inputs, gradient, wrt in
      let wrtInputGradient = Tensor(gradient.value)
      wrtInputGradient.label = "softmax_input_gradient"
      return (wrtInputGradient, Tensor(), Tensor())
    }

    var activationResult: [[[Tensor.Scalar]]] = []

    tensor.value.forEach { d in
      var row: [[Tensor.Scalar]] = []
      d.forEach { r in
        // Vectorized softmax: compute max/sum once for entire row
        let column = calculate(outputs: r)
        row.append(column)
      }
      activationResult.append(row)
    }

    let out = Tensor(activationResult, context: context)
    out.label = type.asString()

    out.setGraph(tensor)

    return out
  }

  /// Vectorized softmax computation - O(n) instead of O(n²)
  private func calculate(outputs: [Tensor.Scalar]) -> [Tensor.Scalar] {
    // Find max once for numerical stability
    let max = outputs.max() ?? 0

    // Compute all exponentials once
    let exps = outputs.map { Tensor.Scalar.exp($0 - max) }

    // Compute sum once
    let sum = exps.reduce(0, +)

    // Normalize all values
    return exps.map { $0 / sum }
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = inputSize
  }
}

