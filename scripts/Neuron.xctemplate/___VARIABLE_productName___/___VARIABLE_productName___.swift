import Foundation
import NumSwift

public final class ___VARIABLE_productName___: Layer {
  public var encodingType: EncodingType
  public var inputSize: TensorSize = TensorSize(array: [0,0,0])
  public var outputSize: TensorSize = TensorSize(array: [0,0,0])
  public var weights: Tensor = Tensor()
  public var biases: Tensor = Tensor()
  public var biasEnabled: Bool = false
  public var trainable: Bool = true
  public var initializer: Initializer?
  public var device: Device = CPU()
  public var isTraining: Bool = true

  public init(inputSize: TensorSize = TensorSize(array: []),
              initializer: InitializerType = .heNormal,
              trainable: Bool = true) {
    self.inputSize = inputSize
    self.initializer = initializer.build()
    self.trainable = trainable
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize,
         biasEnabled,
         outputSize,
         weights,
         biases,
         type
  }
  
  convenience public init(from decoder: Decoder) throws {
    self.init()
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.weights = try container.decodeIfPresent(Tensor.self, forKey: .weights) ?? Tensor()
    self.biases = try container.decodeIfPresent(Tensor.self, forKey: .biases) ?? Tensor()
    self.biasEnabled = try container.decodeIfPresent(Bool.self, forKey: .biasEnabled) ?? false
    self.outputSize = try container.decodeIfPresent(TensorSize.self, forKey: .outputSize) ?? TensorSize(array: [])
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
  
  public func forward(tensor: Tensor) -> Tensor {
    let out = Tensor()
    out.setGraph(tensor)
    return out
  }
  
  public func apply(gradients: (weights: Tensor, biases: Tensor), learningRate: Float) {
    
  }
}
