//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/25/22.
//

import Foundation
import NumSwift

/// Dropout layer for regularization during training
/// Randomly sets input elements to zero with probability `chance` during training
/// Helps prevent overfitting by reducing co-adaptation between neurons
/// During inference, all inputs are passed through (no dropout applied)
public final class Dropout: BaseLayer {
  /// Binary mask tensor indicating which elements to drop
  internal var mask: Tensor = Tensor()
  /// Probability of dropping out each input element (0.0 to 1.0)
  private var chance: Tensor.Scalar
  
  /// Initializes a Dropout layer with specified dropout rate
  /// - Parameters:
  ///   - chance: Dropout probability (0.0 to 1.0). Higher values = more dropout
  ///   - inputSize: Optional input tensor size. Required for first layer in network
  public init(_ chance: Tensor.Scalar, inputSize: TensorSize = TensorSize(array: [])) {
    self.chance = max(min(chance, 1.0), 0.0)
    
    super.init(inputSize: inputSize,
               initializer: nil,
               biasEnabled: false,
               encodingType: .dropout)
    
    self.generateMask()
  }
  
  /// Coding keys for serialization
  enum CodingKeys: String, CodingKey {
    case inputSize,
         chance,
         type,
         mask
  }
  
  /// Initializes Dropout layer from decoder for deserialization
  /// - Parameter decoder: Decoder containing serialized layer data
  /// - Throws: Decoding errors if deserialization fails
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
  
  /// Performs forward pass through the dropout layer
  /// Applies dropout mask during training, passes input unchanged during inference
  /// - Parameters:
  ///   - tensor: Input tensor to apply dropout to
  ///   - context: Network context for computation
  /// - Returns: Output tensor with dropout applied (if training) or unchanged (if inference)
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
  
  /// Applies gradients (no-op for dropout) and generates new dropout mask
  /// Creates a new random mask for the next forward pass
  /// - Parameters:
  ///   - gradients: Gradients (unused for dropout)
  ///   - learningRate: Learning rate (unused for dropout)
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
    mask = Tensor()
    generateMask()
  }
  
  /// Called when input size is set, configures output size and generates mask
  /// For dropout, output size equals input size
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = inputSize
    generateMask()
  }
  
  /// Generates a new random dropout mask
  /// Elements are set to 0 with probability `chance`, otherwise scaled by 1/(1-chance)
  /// The scaling ensures expected output remains unchanged during training
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
