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
  public init(inputSize: TensorSize? = nil,
              linkId: String = UUID().uuidString) {
    super.init(inputSize: inputSize,
               biasEnabled: false,
               linkId: linkId,
               encodingType: .maxPool)
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize,
         type,
         linkId
  }
  
  /// Decodes a MaxPool layer from a serialized model.
  ///
  /// - Parameter decoder: Decoder used during model loading.
  /// - Throws: An error if required values cannot be decoded.
  convenience public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let linkId = try container.decodeIfPresent(String.self, forKey: .linkId) ?? UUID().uuidString
    self.init(linkId: linkId)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
  }
  
  /// Encodes max-pooling layer configuration.
  ///
  /// - Parameter encoder: Encoder used for serialization.
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
    try container.encode(linkId, forKey: .linkId)
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
      let inRows = inputSize.rows
      let inCols = inputSize.columns
      let inSliceSize = inRows * inCols
      let outStorage = TensorStorage.create(count: inSliceSize * inputSize.depth)

      let forwardPooledMaxIndicies = poolingGradients.indicies

      for d in 0..<inputSize.depth {
        let gradPtr = gradient.depthPointer(d)
        let depthBase = outStorage.pointer + d * inSliceSize
        let indicies = forwardPooledMaxIndicies[d]

        for (deltaIdx, index) in indicies.enumerated() {
          depthBase[index.r * inCols + index.c] = gradPtr[deltaIdx]
        }
      }

      return (Tensor(storage: outStorage, size: self.inputSize), Tensor(), Tensor())
    }

    let rows = inputSize.rows
    let columns = inputSize.columns
    let outRows = (rows + 1) / 2
    let outCols = (columns + 1) / 2
    let outSliceSize = outRows * outCols

    var currentIndicies: [[PoolingIndex]] = []
    currentIndicies.reserveCapacity(inputSize.depth)

    let outStorage = TensorStorage.create(count: outSliceSize * inputSize.depth)

    for d in 0..<inputSize.depth {
      let inPtr  = tensor.storage.pointer + d * rows * columns
      let outPtr = outStorage.pointer + d * outSliceSize
      let indices = poolFlat(inputPtr: inPtr, rows: rows, columns: columns, outputPtr: outPtr)
      currentIndicies.append(indices)
    }

    poolingGradients = PoolingGradient(tensorId: tensor.id, indicies: currentIndicies)

    let tensorContext = TensorContext(backpropagate: backwards)
    let outSize = TensorSize(rows: outRows, columns: outCols, depth: inputSize.depth)
    let out = Tensor(storage: outStorage, size: outSize, context: tensorContext)

    out.setGraph(tensor)

    return super.forward(tensor: out, context: context)
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = TensorSize(array: [(inputSize.columns + 1) / 2, (inputSize.rows + 1) / 2, inputSize.depth])
  }
  
  /// MaxPool has no trainable parameters, so this is a no-op.
  ///
  /// - Parameters:
  ///   - gradients: Ignored.
  ///   - learningRate: Ignored.
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
  }
  
  /// Performs 2×2 max pooling for one depth slice, writing results into `outputPtr`.
  /// Returns the flat-index origin of each pooling window's maximum for backpropagation.
  /// `outputPtr` must point to a buffer of size `((rows+1)/2) * ((columns+1)/2)`.
  @discardableResult
  internal func poolFlat(inputPtr: TensorStorage.Pointer,
                         rows: Int,
                         columns: Int,
                         outputPtr: TensorStorage.Pointer) -> [PoolingIndex] {
    var pooledIndicies: [PoolingIndex] = []
    pooledIndicies.reserveCapacity(((rows + 1) / 2) * ((columns + 1) / 2))
    var outIdx = 0

    func safeGet(_ r: Int, _ c: Int) -> Tensor.Scalar {
      guard r >= 0, r < rows, c >= 0, c < columns else { return 0 }
      return inputPtr[r * columns + c]
    }

    for r in stride(from: 0, to: rows, by: 2) {
      for c in stride(from: 0, to: columns, by: 2) {
        let current = safeGet(r, c)
        let right   = safeGet(r + 1, c)
        let bottom  = safeGet(r, c + 1)
        let diag    = safeGet(r + 1, c + 1)

        let candidates: [(Tensor.Scalar, Int, Int)] = [
          (current, r,     c),
          (right,   r + 1, c),
          (bottom,  r,     c + 1),
          (diag,    r + 1, c + 1)
        ]

        let maxVal = Swift.max(Swift.max(Swift.max(current, right), bottom), diag)
        outputPtr[outIdx] = maxVal
        outIdx += 1

        if let first = candidates.first(where: { $0.0 == maxVal }) {
          pooledIndicies.append(PoolingIndex(r: first.1, c: first.2))
        }
      }
    }

    return pooledIndicies
  }
}
