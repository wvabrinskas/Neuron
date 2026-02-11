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
      let wrtInputGradient = Tensor(Tensor.Value(gradient.storage), size: gradient.size)
      wrtInputGradient.label = "softmax_input_gradient"
      return (wrtInputGradient, Tensor(), Tensor())
    }

    let size = tensor.size
    let cols = size.columns
    let rows = size.rows
    let depth = size.depth
    let src = tensor.storage
    var result = Tensor.Value(repeating: 0, count: src.count)

    // Apply softmax per-row (each row of `columns` elements)
    for d in 0..<depth {
      let depthOffset = d * rows * cols
      for r in 0..<rows {
        let rowOffset = depthOffset + r * cols
        
        // Find max for numerical stability
        var rowMax: Tensor.Scalar = src[rowOffset]
        for c in 1..<cols {
          let val = src[rowOffset + c]
          if val > rowMax { rowMax = val }
        }
        
        // Compute exponentials and sum
        var sum: Tensor.Scalar = 0
        for c in 0..<cols {
          let e = Tensor.Scalar.exp(src[rowOffset + c] - rowMax)
          result[rowOffset + c] = e
          sum += e
        }
        
        // Normalize
        for c in 0..<cols {
          result[rowOffset + c] /= sum
        }
      }
    }

    let out = Tensor(result, size: size, context: context)
    out.label = type.asString()

    out.setGraph(tensor)

    return out
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = inputSize
  }
}

