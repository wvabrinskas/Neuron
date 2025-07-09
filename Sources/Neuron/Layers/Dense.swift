//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation
import NumSwift

/// Fully connected (Dense) layer that performs linear transformation
/// Implements the operation: output = input * weights + bias
/// Each input neuron is connected to every output neuron
/// Commonly used for final classification layers and feature transformation
public final class Dense: BaseLayer {
  /// Number of output neurons in this layer
  private var nodes: Int

  /// Initializes a fully connected layer with specified parameters
  /// - Parameters:
  ///   - nodes: Number of output neurons/nodes in this layer
  ///   - inputs: Optional number of input features. Required for first layer
  ///   - initializer: Weight initialization strategy. Default: .heNormal (good for ReLU)
  ///   - biasEnabled: Whether to add bias terms. Default: false
  public init(_ nodes: Int,
              inputs: Int? = nil,
              initializer: InitializerType = .heNormal,
              biasEnabled: Bool = false) {
    
    self.nodes = nodes
    
    super.init(inputSize: nil,
               initializer: initializer,
               biasEnabled: biasEnabled,
               encodingType: .dense)
    
    self.outputSize = TensorSize(array: [nodes, 1, 1])

    if let inputs = inputs {
      inputSize = TensorSize(array: [inputs, 1, 1])
      initializeWeights(inputs: inputs)
    }
  }
  
  /// Coding keys for serialization
  enum CodingKeys: String, CodingKey {
    case biasEnabled,
         inputSize,
         outputSize,
         weights,
         biases,
         type
  }
  
  /// Initializes Dense layer from decoder for deserialization
  /// - Parameter decoder: Decoder containing serialized layer data
  /// - Throws: Decoding errors if deserialization fails
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
  
  
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(weights, forKey: .weights)
    try container.encode(biases, forKey: .biases)
    try container.encode(outputSize, forKey: .outputSize)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
    try container.encode(biasEnabled, forKey: .biasEnabled)
  }
  
  /// Called when input size is set, initializes weights and biases
  /// Dense layers expect flattened input (Nx1x1 dimensions)
  /// Creates weight matrix of size [input_features, output_features]
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    precondition(inputSize.rows == 1 && inputSize.depth == 1, "Dense expects Tensor dimensions of Nx1x1 where N is the columns, got: \(inputSize)")
    
    initializeWeights(inputs: inputSize.columns)
    self.biases = Tensor([Tensor.Scalar](repeating: 0, count: outputSize.depth))
  }
  
  /// Initializes the weight matrix using the specified initialization strategy
  /// Creates connections between all input features and output neurons
  /// - Parameter inputs: Number of input features
  private func initializeWeights(inputs: Int) {
    guard weights.isEmpty else {
      return
    }
    
    var newWeights: [[Tensor.Scalar]] = []
    let outputSizeCount = outputSize.columns
    
    for _ in 0..<outputSizeCount {
      var weightsForNode: [Tensor.Scalar] = []
      for _ in 0..<inputs {
        let w = initializer?.calculate(input: inputs,
                                       out: outputSizeCount) ?? Tensor.Scalar.random(in: -1...1)
        weightsForNode.append(w)
      }
      
      newWeights.append(weightsForNode)
    }
    
    weights = Tensor(newWeights)
  }
  
  /// Performs forward pass through the dense layer
  /// Implements matrix multiplication: output = input * weights + bias
  /// Sets up gradient computation context for backpropagation
  /// - Parameters:
  ///   - tensor: Input tensor (flattened to 1D)
  ///   - context: Network context for computation
  /// - Returns: Output tensor after linear transformation
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    
    let tensorContext = TensorContext { inputs, gradients in
      let gradientsFlat: [Tensor.Scalar] = gradients.value.flatten()
      
      let deltas = self.device.matmul(gradients, self.weights.detached())
      
      let inputsFlat = inputs.value[safe: 0]?[safe: 0] ?? []
      var weightGradients: [[Tensor.Scalar]] = []
      
      for i in 0..<self.nodes {
        let delta = gradientsFlat[i]
        weightGradients.append(inputsFlat * delta)
      }

      return (deltas, Tensor(weightGradients), gradients.sum(axis: -1))
    }
    
    //THIS WAS A MAJOR BUG POINT. DO NOT SWITCH ROWS AND COLUMNS HERE BY ACCIDENT - Billy 05-20-2022
    let weightsTransposed: [[Tensor.Scalar]] = NumSwiftC.tranpose(weights.value[safe: 0] ?? [],
                                                                  size: (rows: outputSize.columns,
                                                                         columns: inputSize.columns))
    
    var dotProducts = device.matmul(tensor, Tensor(weightsTransposed))
    
    if biasEnabled {
      dotProducts = dotProducts + biases.asScalar()
    }
    
    let out = Tensor(dotProducts.value, context: tensorContext)
    out.label = "Dense"
    
    out.setGraph(tensor)

    return out
  }
  
  /// Applies gradients to update layer parameters
  /// Updates weights and biases using computed gradients
  /// - Parameters:
  ///   - gradients: Computed gradients for weights and biases
  ///   - learningRate: Learning rate for parameter updates (unused in this implementation)
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
    weights.value = weights.value - gradients.weights.value
    
    if biasEnabled {
      biases = biases - gradients.biases
    }
  }
}
