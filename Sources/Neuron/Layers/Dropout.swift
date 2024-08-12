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
    
    self.generateMask()
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize,
         chance,
         type,
         mask
  }
  
  convenience public required init(from decoder: Decoder) throws {
    self.init(0)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.chance = try container.decodeIfPresent(Tensor.Scalar.self, forKey: .chance) ?? 0
    self.outputSize = inputSize
    self.mask = try container.decodeIfPresent(Tensor.self, forKey: .mask) ?? Tensor()
  }
  
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(chance, forKey: .chance)
    try container.encode(encodingType, forKey: .type)
    try container.encode(mask, forKey: .mask)
  }
  
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    let context = TensorContext { inputs, gradient in
      let droppedOutGradients = gradient * self.mask
      
      return (droppedOutGradients, Tensor(), Tensor())
    }
    
    var droppedOut = tensor
    
    if trainable {
      droppedOut = tensor * mask
    }

    let out = Tensor(droppedOut.value, context: context)
    
    out.setGraph(tensor)
    
    out.label = String(describing: self)
    
    return out
  }
  
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
    mask = Tensor()
    generateMask()
  }
  
  override public func onInputSizeSet() {
    outputSize = inputSize
    generateMask()
  }
  
  private func generateMask() {
    guard mask.isEmpty else {
      return
    }
    
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
    
    mask = Tensor(maskTensor)
  }
  
}
