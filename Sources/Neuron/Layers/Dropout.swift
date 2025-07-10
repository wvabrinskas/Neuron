//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/25/22.
//

import Foundation
import NumSwift

/// Performs a dropout operation on the inputs based on the chance percentage. The mask changes on every `apply` called.
public final class Dropout: BaseLayer {
  public override var details: String {
    super.details +
    """
    \n
    Chance: \(chance)
    """
  }
  
  internal var mask: Tensor = Tensor()
  private var chance: Tensor.Scalar
  
  /// Default initializer for Dropout layer
  /// - Parameters:
  ///   - chance: Percent change between 0 and 1 of an input node dropping out
  ///   - inputSize: Optional input size at this layer. If this is the first layer you will need to set this.
  public init(_ chance: Tensor.Scalar, inputSize: TensorSize = TensorSize(array: [])) {
    self.chance = max(min(chance, 1.0), 0.0)
    
    super.init(inputSize: inputSize,
               initializer: nil,
               biasEnabled: false,
               encodingType: .dropout)
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize,
         chance,
         type
  }
  
  convenience public required init(from decoder: Decoder) throws {
    self.init(0)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.chance = try container.decodeIfPresent(Tensor.Scalar.self, forKey: .chance) ?? 0
    self.outputSize = inputSize
  }
  
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(chance, forKey: .chance)
    try container.encode(encodingType, forKey: .type)
  }
  
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    let newMask = generateMask()
    
    let context = TensorContext { [newMask] inputs, gradient in
      let outMask = newMask
      let droppedOutGradients = gradient * outMask
      
      return (droppedOutGradients, Tensor(), Tensor())
    }
    
    var droppedOut = tensor
    
    if isTraining && trainable {
      droppedOut = tensor * newMask
    }

    let out = Tensor(droppedOut.value, context: context)
    
    out.setGraph(tensor)
    
    out.label = String(describing: self)
    
    return out
  }
  
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {}
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = inputSize
  }
  
  private func generateMask() -> Tensor {
    var maskTensor: [[[Tensor.Scalar]]] = []

    for _ in 0..<inputSize.depth {
      var row: [[Tensor.Scalar]] = []
      for _ in 0..<inputSize.rows {
        var col: [Tensor.Scalar] = []

        for _ in 0..<inputSize.columns {
          let random = Tensor.Scalar.random(in: 0...1)
          let item: Tensor.Scalar = random <= chance ? 0 : (1 / (1 - chance))
          col.append(item)
        
        }
        row.append(col)
      }
      maskTensor.append(row)
    }
    
    return Tensor(maskTensor)
  }
  
}
