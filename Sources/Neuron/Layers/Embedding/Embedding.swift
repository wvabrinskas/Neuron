import Foundation
import NumSwift

/// An `Embedding` layer that maps each input word vector to a `X` dimensional vector.
public final class Embedding: Layer {
  public var encodingType: EncodingType = .embedding
  public var inputSize: TensorSize = TensorSize(array: [0,0,0])
  public var outputSize: TensorSize = TensorSize(array: [0,0,0])
  public var weights: Tensor = Tensor()
  public var biases: Tensor = Tensor()
  public var biasEnabled: Bool = false
  public var trainable: Bool = true
  public var initializer: Initializer?
  public var device: Device = CPU()
  
  private let inputUnits: Int
  private let vocabSize: Int
  private let batchLength: Int

  public init(inputUnits: Int,
              vocabSize: Int,
              batchLength: Int,
              initializer: InitializerType = .heNormal) {
    let initializerBuilt = initializer.build()
    self.initializer = initializerBuilt
    self.inputUnits = inputUnits
    self.vocabSize = vocabSize
    self.batchLength = batchLength
    self.inputSize = TensorSize(rows: 1,
                                columns: vocabSize,
                                depth: batchLength)
    
    let weights = initializerBuilt.calculate(size: TensorSize(rows: vocabSize,
                                                              columns: inputUnits,
                                                              depth: 1),
                                                input: batchLength * vocabSize,
                                                out: batchLength)
    
    self.weights = weights
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize,
         biasEnabled,
         outputSize,
         weights,
         biases,
         type,
         inputUnits,
         vocabSize,
         batchLength
  }
  
  convenience public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)

    let vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 0
    let inputUnits = try container.decodeIfPresent(Int.self, forKey: .inputUnits) ?? 0
    let batchLength = try container.decodeIfPresent(Int.self, forKey: .batchLength) ?? 0
    
    self.init(inputUnits: inputUnits,
              vocabSize: vocabSize,
              batchLength: batchLength)
    
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
    try container.encode(vocabSize, forKey: .vocabSize)
    try container.encode(batchLength, forKey: .batchLength)
    try container.encode(inputUnits, forKey: .inputUnits)
  }
  
  /// Forward path for the layer
  /// - Parameter tensor: Input word as a 3D tensor with size `rows: 1, columns: vocabSize, depth: batchLength`
  /// - Returns: An output 3D tensor of shape `rows: 1, columns: inputUnits, depth: batchLength`
  public func forward(tensor: Tensor) -> Tensor {
    let context = TensorContext { inputs, gradient in
      var wrtEmbeddings: Tensor = Tensor()

      for i in 0..<gradient.value.count {
        let gradientAtIndex = gradient.value[i]
        
        let embeddingError = Tensor(gradientAtIndex)
        let inputsTransposed = Tensor(inputs.value[i].transpose()) // should only ever have a depth of 1
        let dEmbedding = inputsTransposed.matmul(embeddingError) / Tensor.Scalar(self.batchLength)
        
        if wrtEmbeddings.isEmpty {
          wrtEmbeddings = dEmbedding
        } else {
          let dEmbed = wrtEmbeddings + dEmbedding
          wrtEmbeddings = dEmbed
        }
      }

      // sending the weights through will make them get modified by the optimizer. not sure if that's okay?
      return (Tensor(), wrtEmbeddings)
    }
    
    var out = Tensor(context: context)

    for d in 0..<batchLength {
      let word = Tensor(tensor.value[safe: d] ?? out.value[safe: d] ?? tensor.value[0])
      let getEmbeddings = device.matmul(word, weights.detached())
      
      let concatOut = out.concat(getEmbeddings, axis: 2)
      out = concatOut
    }
    
    out.setGraph(tensor)
    return out
  }
  
  public func apply(gradients: (weights: Tensor, biases: Tensor), learningRate: Float) {
    self.weights = self.weights - gradients.weights
  }
}
