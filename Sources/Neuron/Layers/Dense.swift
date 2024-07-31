//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation
import NumSwift

/// A fully connected layer that performs a `sum(wx) + b` operation using matrix multiplication.
public final class Dense<N: TensorNumeric>: BaseLayer<N> {
  private var nodes: Int

  /// Default initializer for the fully connected layer
  /// - Parameters:
  ///   - nodes: Number of output nodes
  ///   - inputs: Optional input count at this layer. If this is the first layer you will need to set this.
  ///   - initializer: Weight / filter initializer function. Default: `.heNormal`
  ///   - biasEnabled: Boolean defining if the filters have a bias applied. Default: `false`
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
    
    self.weights = try container.decodeIfPresent(Tensor<N>.self, forKey: .weights) ?? Tensor<N>()
    self.biases = try container.decodeIfPresent(Tensor<N>.self, forKey: .biases) ?? Tensor<N>()
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
  
  override public func onInputSizeSet() {
    precondition(inputSize.rows == 1 && inputSize.depth == 1, "Dense expects Tensor<N> dimensions of Nx1x1 where N is the columns, got: \(inputSize)")
    
    initializeWeights(inputs: inputSize.columns)
    self.biases = Tensor<N>([Tensor<N>.Scalar](repeating: 0, count: outputSize.depth))
  }
  
  private func initializeWeights(inputs: Int) {
    guard weights.isEmpty else {
      return
    }
    
    var newWeights: [[Tensor<N>.Scalar]] = []
    let outputSizeCount = outputSize.columns
    
    for _ in 0..<outputSizeCount {
      var weightsForNode: [Tensor<N>.Scalar] = []
      for _ in 0..<inputs {
        let w = initializer?.calculate(input: inputs,
                                       out: outputSizeCount) ?? Tensor<N>.Scalar.random(in: -1...1)
        weightsForNode.append(w)
      }
      
      newWeights.append(weightsForNode)
    }
    
    weights = Tensor<N>(newWeights)
  }
  
  public override func forward(tensor: Tensor<N>) -> Tensor<N> {
    
    let tensorContext = TensorContext<N> { inputs, gradients in
      let gradientsFlat: [Tensor<N>.Scalar] = gradients.value.flatten()
      
      let deltas = self.device.matmul(gradients, self.weights.detached())
      
      let inputsFlat = inputs.value[safe: 0]?[safe: 0] ?? []
      var weightGradients: [[Tensor<N>.Scalar]] = []
      
      for i in 0..<self.nodes {
        let delta = gradientsFlat[i]
        weightGradients.append(inputsFlat * delta)
      }

      return (deltas, Tensor<N>(weightGradients), gradients.sum(axis: -1))
    }
    
    //THIS WAS A MAJOR BUG POINT. DO NOT SWITCH ROWS AND COLUMNS HERE BY ACCIDENT - Billy 05-20-2022
    let weightsTransposed: [[Tensor<N>.Scalar]] = NumSwiftC.tranpose(weights.value[safe: 0] ?? [],
                                                                  size: (rows: outputSize.columns,
                                                                         columns: inputSize.columns))
    
    var dotProducts = device.matmul(tensor, Tensor<N>(weightsTransposed))
    
    if biasEnabled {
      dotProducts = dotProducts + biases.asScalar()
    }
    
    let out = Tensor<N>(dotProducts.value, context: tensorContext)
    out.label = "Dense"
    
    out.setGraph(tensor)

    return out
  }
  
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor<N>.Scalar) {
    weights.value = weights.value - gradients.weights.value
    
    if biasEnabled {
      biases = biases - gradients.biases
    }
  }
}
