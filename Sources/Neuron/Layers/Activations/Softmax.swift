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
  public init(inputSize: TensorSize = TensorSize(array: [])) {
    super.init(inputSize: inputSize,
               type: .softmax,
               encodingType: .softmax)
  }
  
  public override func forward(tensor: Tensor) -> Tensor {
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
  
  private func calculate(index: Int, outputs: [Tensor.Scalar]) -> Tensor.Scalar {
    let max = outputs.max() ?? 1
    var sum: Tensor.Scalar = 0
    outputs.forEach { (output) in
      sum += Tensor.Scalar.pow(Tensor.Scalar(Darwin.M_E), output - max)
    }
    
    return Tensor.Scalar.pow(Tensor.Scalar(Darwin.M_E), outputs[index] - max) / sum
  }
  
  override public func onInputSizeSet() {
    outputSize = inputSize
  }
}

