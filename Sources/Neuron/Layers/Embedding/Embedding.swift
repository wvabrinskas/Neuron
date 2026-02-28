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
              trainable: Bool = false,
              linkId: String = UUID().uuidString) {
    self.inputUnits = inputUnits
    self.vocabSize = vocabSize
    self.batchLength = batchLength
    
    super.init(inputSize: TensorSize(rows: 1,
                                     columns: 1,
                                     depth: batchLength),
               initializer: initializer, biasEnabled: false,
               linkId: linkId,
               encodingType: .embedding)
    
    self.outputSize = TensorSize(rows: 1, columns: inputUnits, depth: batchLength)

    let weights = initializer.build().calculate(size: TensorSize(rows: 1,
                                                                 columns: inputUnits,
                                                                 depth: vocabSize),
                                                input: batchLength * inputUnits,
                                                out: inputUnits * vocabSize)
    
    self.weights = weights
    self.weights.label = "Embedding weights"

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
         batchLength,
         linkId
  }
  
  convenience required public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)

    let vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 0
    let inputUnits = try container.decodeIfPresent(Int.self, forKey: .inputUnits) ?? 0
    let batchLength = try container.decodeIfPresent(Int.self, forKey: .batchLength) ?? 0
    let linkId = try container.decodeIfPresent(String.self, forKey: .linkId) ?? UUID().uuidString
    
    self.init(inputUnits: inputUnits,
              vocabSize: vocabSize,
              batchLength: batchLength,
              linkId: linkId)
    
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.weights = try container.decodeIfPresent(Tensor.self, forKey: .weights) ?? Tensor()
    self.biases = try container.decodeIfPresent(Tensor.self, forKey: .biases) ?? Tensor()
    self.biasEnabled = try container.decodeIfPresent(Bool.self, forKey: .biasEnabled) ?? false
    self.outputSize = try container.decodeIfPresent(TensorSize.self, forKey: .outputSize) ?? TensorSize(array: [])
  }
  
  /// Encodes embedding parameters and shape metadata.
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
    try container.encode(vocabSize, forKey: .vocabSize)
    try container.encode(batchLength, forKey: .batchLength)
    try container.encode(inputUnits, forKey: .inputUnits)
    try container.encode(linkId, forKey: .linkId)
  }
  
  /// Forward path for the layer
  /// - Parameter tensor: Input word as a 3D tensor with size `rows: 1, columns: vocabSize, depth: batchLength`
  /// - Returns: An output 3D tensor of shape `rows: 1, columns: inputUnits, depth: batchLength`
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    var indicies: [Int] = []
    
    let tensorContext = TensorContext { [inputUnits, vocabSize] inputs, gradient, wrt in
      let sliceSize = inputUnits
      let embSize = TensorSize(rows: 1, columns: inputUnits, depth: vocabSize)
      var embStorage = Tensor.Value(repeating: 0, count: sliceSize * vocabSize)
      
      for (i, index) in indicies.enumerated() {
        guard index < vocabSize, i < gradient.size.depth else { continue }
        let gradSlice = gradient.depthSlice(i)
        let offset = index * sliceSize
        for j in 0..<min(gradSlice.count, sliceSize) {
          embStorage[offset + j] += gradSlice[j]
        }
      }

      let result = Tensor(embStorage, size: embSize)
      result.label = "Embedding gradients"
      
      return (Tensor(), result, Tensor())
    }
    
    // Build output by looking up each input's embedding from weights
    let weightDepth = weights.size.depth
    let sliceSize = weights.size.rows * weights.size.columns
    var outSlices = [Tensor.Value]()
    
    for d in 0..<tensor.size.depth {
      let slice = tensor.depthSlice(d)
      let scalar = slice.isEmpty ? Tensor.Scalar(0) : slice[0]
      let index = Int(scalar)
      indicies.append(index)
      
      guard index >= 0 && index < weightDepth else {
        fatalError("Could not find embedding for index: \(index)")
      }
      
      outSlices.append(weights.depthSlice(index))
    }
    
    // Assemble output tensor
    var outStorage = Tensor.Value()
    outStorage.reserveCapacity(sliceSize * outSlices.count)
    outSlices.forEach { outStorage.append(contentsOf: $0) }
    
    let outSize = TensorSize(rows: weights.size.rows, columns: weights.size.columns, depth: outSlices.count)
    let out = Tensor(outStorage, size: outSize, context: tensorContext)
    
    out.setGraph(tensor)

    return super.forward(tensor: out, context: context)
  }
  
  /// Applies embedding weight updates from optimizer gradients.
  ///
  /// - Parameters:
  ///   - gradients: Gradients with embedding-weight derivatives in `weights`.
  ///   - learningRate: Learning rate used when `usesOptimizer == false`.
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
    if trainable {
      if usesOptimizer {
        weights = weights.copy() - gradients.weights
      } else {
        weights = weights.copy() - learningRate * gradients.weights
      }
    }

    weights.label = "Embedding weights"
  }
}
