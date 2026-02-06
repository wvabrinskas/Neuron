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
  
  internal var poolingGradients: [PoolingGradient] = []
  private lazy var queue: OperationQueue = OperationQueue()
  
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
  
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
  }
  
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    func backwards(input: Tensor, gradient: Tensor, wrt: Tensor?) -> (Tensor, Tensor, Tensor) {
      var outStorage = ContiguousArray<Tensor.Scalar>(repeating: 0,
                                                       count: self.inputSize.rows * self.inputSize.columns * self.inputSize.depth)
      
      // operation is performed first then returned
      queue.addSynchronousOperation { [weak self] in
        guard let sSelf = self else { return }
        
        guard let forwardPooledMaxIndicies = sSelf.poolingGradients.first(where: { $0.tensorId == input.id })?.indicies else {
          return
        }
        
        let inRows = sSelf.inputSize.rows
        let inCols = sSelf.inputSize.columns
        let gradCols = gradient._size.columns
        
        for d in 0..<sSelf.inputSize.depth {
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
      }
      
      if outStorage.allSatisfy({ $0 == 0 }) && !gradient.isEmpty {
        return (gradient, Tensor(), Tensor())
      }
      
      return (Tensor(outStorage, size: self.inputSize), Tensor(), Tensor())
    }
    
    let rows = inputSize.rows
    let columns = inputSize.columns
    let outRows = (rows + 1) / 2
    let outCols = (columns + 1) / 2
    
    var currentIndicies: [[PoolingIndex]] = []
    currentIndicies.reserveCapacity(inputSize.depth)
    
    var outStorage = ContiguousArray<Tensor.Scalar>(repeating: 0, count: outRows * outCols * inputSize.depth)

    for d in 0..<inputSize.depth {
      let slice = tensor.depthSlice(d)
      let (poolResult, indices) = poolFlat(input: slice, rows: rows, columns: columns)
      currentIndicies.append(indices)
      
      let depthOffset = d * outRows * outCols
      for j in 0..<poolResult.count {
        outStorage[depthOffset + j] = poolResult[j]
      }
    }

    queue.addBarrierBlock {
      self.setGradients(indicies: currentIndicies, id: tensor.id)
    }
          
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
    self.poolingGradients.append(PoolingGradient(tensorId: id, indicies: indicies))
  }
  
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
    poolingGradients.removeAll(keepingCapacity: true)
  }
  
  internal func poolFlat(input: ContiguousArray<Tensor.Scalar>, rows: Int, columns: Int) -> (ContiguousArray<Tensor.Scalar>, [PoolingIndex]) {
    var results = ContiguousArray<Tensor.Scalar>()
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
