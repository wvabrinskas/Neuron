import Foundation
import NumSwift

/// An `Embedding` layer that maps each input word vector to a `X` dimensional vector.
public final class Embedding: BaseLayer {
  private let inputUnits: Int
  private let vocabSize: Int
  private let batchLength: Int
  
  
  /// Default initializer
  /// - Parameters:
  ///   - inputUnits: Number of hidden neurons in the dense layer
  ///   - vocabSize: Size of the vocabulary
  ///   - batchLength: Length of the input vector
  ///   - initializer: Weight initializer
  ///   - trainable: Whether or not to update weights
  public init(inputUnits: Int,
              vocabSize: Int,
              batchLength: Int,
              initializer: InitializerType = .xavierNormal,
              trainable: Bool = false) {
    self.inputUnits = inputUnits
    self.vocabSize = vocabSize
    self.batchLength = batchLength
    
    super.init(inputSize: TensorSize(rows: 1,
                                     columns: 1,
                                     depth: batchLength),
               initializer: initializer, biasEnabled: false,
               encodingType: .embedding)
    
    self.outputSize = TensorSize(rows: 1, columns: inputUnits, depth: batchLength)

    let weights = initializer.build().calculate(size: TensorSize(rows: 1,
                                                                 columns: inputUnits,
                                                                 depth: vocabSize),
                                                input: batchLength * inputUnits,
                                                out: inputUnits * vocabSize)
    
    self.weights = weights
    // manages its own weight updates
    self.usesOptimizer = true
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
  
  convenience required public init(from decoder: Decoder) throws {
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
  
  public override func encode(to encoder: Encoder) throws {
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
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    var indicies: [Int] = []
    let weightShape = weights.shape
    
    let context = TensorContext { inputs, gradient, wrt in
      var wrtEmbeddings: Tensor.Data = Tensor.fillWith(value: 0, size: TensorSize(array: weightShape)).value
            
      for (i, index) in indicies.enumerated() {
        if let current = wrtEmbeddings[safe: index] {
          wrtEmbeddings[index] = current + gradient.value[safe: i, [[0]]]
        } else {
          wrtEmbeddings[index] = gradient.value[safe: i, [[0]]]
        }
      }

      let result = Tensor(wrtEmbeddings)
      
      result.label = "Embedding gradients"
      
      return (Tensor(), result, Tensor())
    }
    
    var outValue: Tensor.Data = []
    
    for sequence in tensor.value {
      let flat = Tensor(sequence).asScalar()
      let index = Int(flat)
      indicies.append(index)
      
      guard let lookup = weights.value[safe: index] else {
        fatalError()
      }
      
      outValue.append(lookup)
    }
    
    let out = Tensor(outValue, context: context)
    
    out.label = String(describing: self)
  
    out.setGraph(tensor)

    return out
  }
  
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
    if trainable {
      if usesOptimizer {
        weights = weights.copy() - gradients.weights
      } else {
        weights = weights.copy() - learningRate * gradients.weights
      }
    }
  }
}
