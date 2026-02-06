//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/22/22.
//

import Foundation
import NumSwift

/// Performs a layer normalization function.
public final class LayerNormalize: BaseLayer {
  public override var weights: Tensor {
    get {
      // For printing purposes. Not actually used
      beta.concat(gamma, axis: 2)
    }
    set {}
  }

  private var epsilon: Tensor.Scalar
  public var gamma: Tensor = .init()
  public var beta: Tensor = .init()
  @Atomic private var iterations: Int = 0

  public enum CodingKeys: String, CodingKey {
    case gamma, beta, epsilon, inputSize
  }
  
  /// Default initializer for layer normalization.
  /// - Parameters:
  ///   - epsilon: Epsilon value for normalization. Defualt: `1e-10`
  ///   - gamma: Gamme value for normalization
  ///   - beta: Beta value for normalization
  ///   - inputSize: Optional input size at this layer. If this is the first layer you will need to set this.
  public init(epsilon: Tensor.Scalar = .stabilityFactor,
              gamma: Tensor = .init(),
              beta: Tensor = .init(),
              inputSize: TensorSize? = nil) {
    self.epsilon = epsilon
    self.beta = beta
    self.gamma = gamma
    
    super.init(inputSize: inputSize,
               biasEnabled: false,
               encodingType: .layerNormalize)
    
    if let inputSize {
      self.outputSize = inputSize
    }
    
    setupTrainables()
  }
  
  convenience public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let gamma = try container.decodeIfPresent(Tensor.self, forKey: .gamma) ?? .init()
    let beta = try container.decodeIfPresent(Tensor.self, forKey: .beta) ?? .init()
    let epsilon = try container.decodeIfPresent(Tensor.Scalar.self, forKey: .epsilon) ?? .stabilityFactor

    self.init(epsilon: epsilon,
              gamma: gamma,
              beta: beta)
    
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.outputSize = inputSize
    
    setupTrainables()
  }
  
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(beta, forKey: .beta)
    try container.encode(gamma, forKey: .gamma)
    try container.encode(epsilon, forKey: .epsilon)
  }
  
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    let context = TensorContext { inputs, gradient, wrt in
      self.backwardFlat(inputs: inputs, gradient: gradient)
    }
    
    let forwardStorage = normalizeFlat(inputs: tensor)
    let out = Tensor(forwardStorage, size: tensor.size, context: context)
    out.setGraph(tensor)
    return out
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = inputSize
    setupTrainables()
  }

  private func normalizeFlat(inputs: Tensor) -> ContiguousArray<Tensor.Scalar> {
    let depth = inputs.depthSliceCount
    let sliceSize = inputSize.rows * inputSize.columns
    let total = Tensor.Scalar(sliceSize)
    var outStorage = ContiguousArray<Tensor.Scalar>(repeating: 0, count: inputs.storage.count)
    
    for i in 0..<depth {
      let slice = inputs.depthSlice(i)
      let mean = NumSwiftFlat.mean(slice)
      let centered = NumSwiftFlat.subtract(slice, scalar: mean)
      let variance = NumSwiftFlat.sumOfSquares(centered) / total
      let std = Tensor.Scalar.sqrt(variance + epsilon)
      
      let normalized = NumSwiftFlat.divide(centered, scalar: std)
      // result = normalized * gamma[i] + beta[i]
      let gammaSlice = gamma.depthSlice(i)
      let betaSlice = beta.depthSlice(i)
      let scaled = NumSwiftFlat.add(NumSwiftFlat.multiply(normalized, gammaSlice), betaSlice)
      
      let offset = i * sliceSize
      for j in 0..<sliceSize { outStorage[offset + j] = scaled[j] }
    }
    
    return outStorage
  }
  
  private func backwardFlat(inputs: Tensor, gradient: Tensor) -> (input: Tensor, weight: Tensor, bias: Tensor) {
    let depth = inputs.depthSliceCount
    let sliceSize = inputSize.rows * inputSize.columns
    
    // We use Tensor operations per-depth for the complex backward math
    // but construct depth-1 Tensors from flat slices instead of going through .value
    var dInputSlices = [ContiguousArray<Tensor.Scalar>]()
    var dGammaSlices = [ContiguousArray<Tensor.Scalar>]()
    var dBetaSlices = [ContiguousArray<Tensor.Scalar>]()
    
    let sliceShape = TensorSize(rows: inputSize.rows, columns: inputSize.columns, depth: 1)
    
    for i in 0..<depth {
      let featureTensor = inputs.depthSliceTensor(i)
      let gradTensor = gradient.depthSliceTensor(i)
      let gammaTensor = gamma.depthSliceTensor(i)
      
      let N = Tensor.Scalar(inputSize.rows)
      
      let mean = featureTensor.mean(axis: 1)
      let variance = featureTensor.variance(axis: 1)
      let varianceEpsilon = variance + epsilon
      let std = varianceEpsilon.sqrt()
      
      let inputsMinusMean = featureTensor - mean
      let x_norm = inputsMinusMean / std
      
      let dL_dbeta = gradTensor.sum(axis: 2)
      let dL_dgamma = (x_norm * gradTensor).sum(axis: 2)
      
      let invNStd = gammaTensor * (Tensor.Scalar(1) / (N * std))
      let line2 = N * gradTensor
      let line3 = dL_dbeta
      let line4 = inputsMinusMean / varianceEpsilon
      let line5 = (featureTensor - gradTensor * mean).sum(axis: 2)
      
      let dl_dx = invNStd * (line2 - line3 - line4 * line5)
      
      dInputSlices.append(dl_dx.storage)
      dGammaSlices.append(dL_dgamma.storage)
      dBetaSlices.append(dL_dbeta.storage)
    }
    
    // Assemble full tensors from per-depth slices
    var dInputStorage = ContiguousArray<Tensor.Scalar>()
    dInputSlices.forEach { dInputStorage.append(contentsOf: $0) }
    
    var dGammaStorage = ContiguousArray<Tensor.Scalar>()
    dGammaSlices.forEach { dGammaStorage.append(contentsOf: $0) }
    
    var dBetaStorage = ContiguousArray<Tensor.Scalar>()
    dBetaSlices.forEach { dBetaStorage.append(contentsOf: $0) }
    
    let dGammaTensor = Tensor(dGammaStorage, size: TensorSize(rows: inputSize.rows, columns: inputSize.columns, depth: depth))
    let dBetaTensor = Tensor(dBetaStorage, size: TensorSize(rows: inputSize.rows, columns: inputSize.columns, depth: depth))
    
    return (Tensor(dInputStorage, size: inputs.size),
            dGammaTensor.concat(dBetaTensor, axis: 2),
            Tensor())
  }
  
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
    let gammaWeights = gradients.weights[0..., 0..., 0..<inputSize.depth]
    let betaWeights = gradients.weights[0..., 0..., inputSize.depth...]

    gamma = gamma - gammaWeights
    beta = beta - betaWeights
  }
  
  private func setupTrainables() {
    if gamma.isEmpty {
      self.gamma = Tensor(NumSwift.onesLike((rows: inputSize.rows, columns: inputSize.columns, depth: inputSize.depth)))
    }
    
    if beta.isEmpty {
      self.beta = Tensor(NumSwift.zerosLike((rows: inputSize.rows, columns: inputSize.columns, depth: inputSize.depth)))
    }
  }
}
