//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation
import NumSwift

/// A fully connected layer that performs a `sum(wx) + b` operation using matrix multiplication.
public final class Dense: Layer {
  public var encodingType: EncodingType = .dense
  public var device: Device = CPU()
  public var biasEnabled: Bool = true
  public var trainable: Bool = true
  public var inputSize: TensorSize = TensorSize(array: []) {
    didSet {
      initializeWeights(inputs: inputSize.columns)
    }
  }
  public private(set) var initializer: Initializer?
  public let outputSize: TensorSize
  public internal(set) var weights: Tensor = Tensor()
  public private(set)var biases: Tensor = Tensor()
  public var isTraining: Bool = true

  /// Default initializer for the fully connected layer
  /// - Parameters:
  ///   - nodes: Number of output nodes
  ///   - inputs: Optional input count at this layer. If this is the first layer you will need to set this.
  ///   - initializer: Weight / filter initializer function. Default: `.heNormal`
  ///   - biasEnabled: Boolean defining if the filters have a bias applied. Default: `false`
  public init(_ nodes: Int,
              inputs: Int? = nil,
              initializer: InitializerType = .heNormal,
              biasEnabled: Bool = true) {
    outputSize = TensorSize(array: [nodes, 1, 1])
    
    if let inputs = inputs {
      inputSize = TensorSize(array: [inputs, 1, 1])
      initializeWeights(inputs: inputs)
    }
    
    self.initializer = initializer.build()
    self.biases = Tensor([Tensor.Scalar](repeating: 0, count: outputSize.depth))
    self.biasEnabled = biasEnabled
  }
  
  enum CodingKeys: String, CodingKey {
    case biasEnabled,
         inputSize,
         outputSize,
         weights,
         biases,
         type
  }
  
  convenience public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let outputSize = try container.decodeIfPresent(TensorSize.self, forKey: .outputSize) ?? TensorSize(array: [])
    let inputs = outputSize.columns
    self.init(inputs)
    
    self.weights = try container.decodeIfPresent(Tensor.self, forKey: .weights) ?? Tensor()
    self.biases = try container.decodeIfPresent(Tensor.self, forKey: .biases) ?? Tensor()
    self.biasEnabled = try container.decodeIfPresent(Bool.self, forKey: .biasEnabled) ?? false
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
  }
  
  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(weights, forKey: .weights)
    try container.encode(biases, forKey: .biases)
    try container.encode(outputSize, forKey: .outputSize)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
    try container.encode(biasEnabled, forKey: .biasEnabled)
  }
  
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
                                       out: outputSizeCount) ?? Float.random(in: -1...1)
        weightsForNode.append(w)
      }
      
      newWeights.append(weightsForNode)
    }
    
    weights = Tensor(newWeights)
  }
  
  public func forward(tensor: Tensor) -> Tensor {
    
    let tensorContext = TensorContext { inputs, gradients in
      let gradientsFlat: [Tensor.Scalar] = gradients.value.flatten()
      
      let deltas = self.device.matmul(gradients, self.weights.detached())
      
      let inputsFlat = inputs.value[safe: 0]?[safe: 0] ?? []
      var weightGradients: [[Tensor.Scalar]] = []
      
      for i in 0..<gradientsFlat.count {
        let delta = gradientsFlat[i]
        weightGradients.append(inputsFlat * delta)
      }

      return (deltas, Tensor(weightGradients), gradients.sum(axis: -1))
    }
    
    //THIS WAS A MAJOR BUG POINT. DO NOT SWITCH ROWS AND COLUMNS HERE BY ACCIDENT - Billy 05-20-2022
    let weightsTransposed: [[Tensor.Scalar]] = weights.value.flatten().transpose(columns: inputSize.columns,
                                                                               rows: outputSize.columns)
                                                                       .reshape(columns: outputSize.columns)
    
    var dotProducts = device.matmul(tensor, Tensor(weightsTransposed))
    
    if biasEnabled {
      dotProducts = dotProducts + biases.asScalar()
    }
    
    let out = Tensor(dotProducts.value, context: tensorContext)
    out.label = "Dense"
    
    out.setGraph(tensor)

    return out
  }
  
  public func apply(gradients: Optimizer.Gradient, learningRate: Float) {
    weights.value = weights.value - gradients.weights.value
    
    if biasEnabled {
      biases = biases - gradients.biases
    }
  }
}
