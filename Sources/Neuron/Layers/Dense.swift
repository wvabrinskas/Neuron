//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation
import NumSwift

/// A fully connected layer that performs a `sum(wx) + b` operation using matrix multiplication.
public final class Dense: BaseLayer {
  private var nodes: Int

  /// Default initializer for the fully connected layer
  /// - Parameters:
  ///   - nodes: Number of output nodes
  ///   - inputs: Optional input count at this layer. If this is the first layer you will need to set this.
  ///   - initializer: Weight / filter initializer function. Default: `.heNormal`
  ///   - biasEnabled: Boolean defining if the filters have a bias applied. Default: `false`
  ///   - linkId: Set this to reference the output of this layer in an arithmetic layer. eg a Shortcut path
  public init(_ nodes: Int,
              inputs: Int? = nil,
              initializer: InitializerType = .heNormal,
              biasEnabled: Bool = false,
              linkId: String = UUID().uuidString) {
    
    self.nodes = nodes
    
    super.init(inputSize: nil,
               initializer: initializer,
               biasEnabled: biasEnabled,
               linkId: linkId,
               encodingType: .dense)
    
    self.outputSize = TensorSize(array: [nodes, 1, 1])

    if let inputs = inputs {
      inputSize = TensorSize(array: [inputs, 1, 1])
      initializeWeights(inputs: inputs)
    }
  }
  
  enum CodingKeys: String, CodingKey {
    case biasEnabled,
         inputSize,
         outputSize,
         weights,
         biases,
         type
  }
  
  convenience public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let outputSize = try container.decodeIfPresent(TensorSize.self, forKey: .outputSize) ?? TensorSize(array: [])
    let inputs = outputSize.columns
    self.init(inputs)
    
    self.weights = try container.decodeIfPresent(Tensor.self, forKey: .weights) ?? Tensor()
    self.biases = try container.decodeIfPresent(Tensor.self, forKey: .biases) ?? Tensor()
    self.biasEnabled = try container.decodeIfPresent(Bool.self, forKey: .biasEnabled) ?? false
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
  }
  
  
  /// Encodes dense layer parameters for persistence.
  ///
  /// - Parameter encoder: Encoder used for serialization.
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(weights, forKey: .weights)
    try container.encode(biases, forKey: .biases)
    try container.encode(outputSize, forKey: .outputSize)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
    try container.encode(biasEnabled, forKey: .biasEnabled)
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    precondition(inputSize.rows == 1 && inputSize.depth == 1, "Dense expects Tensor dimensions of Nx1x1 where N is the columns, got: \(inputSize)")
    
    initializeWeights(inputs: inputSize.columns)
    self.biases = Tensor([Tensor.Scalar](repeating: 0, count: outputSize.columns))
  }
  
  private func initializeWeights(inputs: Int) {
    guard weights.isEmpty else {
      return
    }
    
    let outputSizeCount = outputSize.columns
    
    weights = initializer.calculate(size: .init(rows: outputSizeCount,
                                                columns: inputs,
                                                depth: 1),
                                    input: inputs, out: outputSizeCount)
  }
  
  /// Computes dense affine transformation `xW^T + b`.
  ///
  /// - Parameters:
  ///   - tensor: Input tensor.
  ///   - context: Network execution context.
  /// - Returns: Dense output tensor with attached gradient context.
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    
    let tensorContext = TensorContext { inputs, gradients, wrt in
      let deltas = self.device.matmul(gradients, self.weights.detached())

      let weightGradients = self.device.matmul(gradients.transposed(), inputs)

      return (deltas, weightGradients, gradients)
    }
    
    //THIS WAS A MAJOR BUG POINT. DO NOT SWITCH ROWS AND COLUMNS HERE BY ACCIDENT - Billy 05-20-2022
    let weightsTransposed = weights.transposed()
    
    var dotProducts = device.matmul(tensor, weightsTransposed)
    
    if biasEnabled {
      dotProducts = dotProducts.copy() + biases
    }
    
    let out = Tensor(dotProducts.storage, size: dotProducts.size, context: tensorContext)
    
    out.setGraph(tensor)

    return out
  }
  
  /// Applies dense weight and optional bias updates.
  ///
  /// - Parameters:
  ///   - gradients: Weight and bias gradients for this layer.
  ///   - learningRate: Learning rate already reflected in optimizer gradient output.
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
    weights = weights.copy() - gradients.weights

    if biasEnabled {
      biases = biases.copy() - gradients.biases
    }
  }
}
