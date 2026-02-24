//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/5/22.
//

import Foundation
import NumSwift

/// Will decrease the size of the input tensor by half using a max pooling technique.
public final class MaxPool: BaseLayer {
  internal struct PoolingIndex: Hashable, Codable {
    var r: Int
    var c: Int
  }
  
  internal struct PoolingGradient: Hashable, Codable {
    static func == (lhs: MaxPool.PoolingGradient, rhs: MaxPool.PoolingGradient) -> Bool {
      lhs.tensorId == rhs.tensorId
    }
    
    var tensorId: Tensor.ID
    var indicies: [[PoolingIndex]]
  }
    
  /// Default initializer for max pooling.
  /// - Parameter inputSize: Optional input size at this layer. If this is the first layer you will need to set this.
  public init(inputSize: TensorSize? = nil) {
    super.init(inputSize: inputSize,
               biasEnabled: false,
               encodingType: .maxPool)
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize,
         type
  }
  
  convenience public required init(from decoder: Decoder) throws {
    self.init()
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
  }
  
  /// Encodes max-pooling layer configuration.
  ///
  /// - Parameter encoder: Encoder used for serialization.
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
  }
  
  /// Performs 2x2 max pooling on each depth slice.
  ///
  /// - Parameters:
  ///   - tensor: Input tensor.
  ///   - context: Network execution context.
  /// - Returns: Pooled tensor with routing indices captured for backpropagation.
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    var poolingGradients: PoolingGradient = .init(tensorId: tensor.id, indicies: [])
    
    func backwards(input: Tensor, gradient: Tensor, wrt: Tensor?) -> (Tensor, Tensor, Tensor) {
      var outStorage = Tensor.Value(repeating: 0, count: self.inputSize.rows * self.inputSize.columns * self.inputSize.depth)
    
      // operation is performed first then returned
      
      let forwardPooledMaxIndicies = poolingGradients.indicies
      
      let inRows = inputSize.rows
      let inCols = inputSize.columns
      let gradCols = gradient.size.columns
      
      for d in 0..<inputSize.depth {
        let gradSlice = gradient.depthSlice(d)
        var deltaIdx = 0
        let indicies = forwardPooledMaxIndicies[d]
        let depthOffset = d * inRows * inCols
        
        for index in indicies {
          if deltaIdx < gradSlice.count {
            outStorage[depthOffset + index.r * inCols + index.c] = gradSlice[deltaIdx]
            deltaIdx += 1
          }
        }
      }

     // print(outStorage.count, inputSize.columns * inputSize.rows * inputSize.depth)
      return (Tensor(outStorage, size: self.inputSize), Tensor(), Tensor())
    }
    
    let rows = inputSize.rows
    let columns = inputSize.columns
    let outRows = (rows + 1) / 2
    let outCols = (columns + 1) / 2
    
    var currentIndicies: [[PoolingIndex]] = []
    currentIndicies.reserveCapacity(inputSize.depth)
    
    var outStorage = Tensor.Value(repeating: 0, count: outRows * outCols * inputSize.depth)

    for d in 0..<inputSize.depth {
      let slice = tensor.depthSlice(d)
      let (poolResult, indices) = poolFlat(input: slice, rows: rows, columns: columns)
      currentIndicies.append(indices)
      
      let depthOffset = d * outRows * outCols
      for j in 0..<poolResult.count {
        outStorage[depthOffset + j] = poolResult[j]
      }
    }

    poolingGradients = PoolingGradient(tensorId: tensor.id, indicies: currentIndicies)

    let context = TensorContext(backpropagate: backwards)
    let outSize = TensorSize(rows: outRows, columns: outCols, depth: inputSize.depth)
    let out = Tensor(outStorage, size: outSize, context: context)
    
    out.setGraph(tensor)
    
    return out
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = TensorSize(array: [(inputSize.columns + 1) / 2, (inputSize.rows + 1) / 2, inputSize.depth])
  }
  
  private func setGradients(indicies: [[PoolingIndex]], id: Tensor.ID) {
  }
  
  /// MaxPool has no trainable parameters, so this is a no-op.
  ///
  /// - Parameters:
  ///   - gradients: Ignored.
  ///   - learningRate: Ignored.
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
  }
  
  internal func poolFlat(input: Tensor.Value, rows: Int, columns: Int) -> (Tensor.Value, [PoolingIndex]) {
    var results = Tensor.Value()
    var pooledIndicies: [PoolingIndex] = []
    
    func safeGet(_ r: Int, _ c: Int) -> Tensor.Scalar {
      guard r >= 0, r < rows, c >= 0, c < columns else { return 0 }
      return input[r * columns + c]
    }
    
    for r in stride(from: 0, to: rows, by: 2) {
      for c in stride(from: 0, to: columns, by: 2) {
        let current = safeGet(r, c)
        let right = safeGet(r + 1, c)
        let bottom = safeGet(r, c + 1)
        let diag = safeGet(r + 1, c + 1)
        
        let indiciesToCheck = [(current, r, c),
                               (right, r + 1, c),
                               (bottom, r, c + 1),
                               (diag, r + 1, c + 1)]
        
        let maxVal = Swift.max(Swift.max(Swift.max(current, right), bottom), diag)
        if let firstIndicies = indiciesToCheck.first(where: { $0.0 == maxVal }) {
          pooledIndicies.append(PoolingIndex(r: firstIndicies.1, c: firstIndicies.2))
        }
        results.append(maxVal)
      }
    }
    
    return (results, pooledIndicies)
  }
}
