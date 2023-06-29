//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/5/22.
//

import Foundation
import NumSwift

/// Will decrease the size of the input tensor by half using a max pooling technique.
public final class MaxPool: Layer {
  public var encodingType: EncodingType = .maxPool
  public var device: Device = CPU()
  public var biasEnabled: Bool = true

  internal struct PoolingIndex: Hashable, Codable {
    var r: Int
    var c: Int
  }
  
  internal struct PoolingGradient: Hashable, Codable {
    static func == (lhs: MaxPool.PoolingGradient, rhs: MaxPool.PoolingGradient) -> Bool {
      lhs.tensorId == rhs.tensorId
    }
    
    var tensorId: UUID
    var indicies: [[PoolingIndex]]
  }
  
  public var inputSize: TensorSize = TensorSize(array: [])
  public var outputSize: TensorSize {
    TensorSize(array: [inputSize.columns / 2, inputSize.rows / 2, inputSize.depth])
  }
  
  public var weights: Tensor = Tensor()
  public var biases: Tensor = Tensor()
  public var trainable: Bool = true
  public var initializer: Initializer?
  internal var poolingGradients: [PoolingGradient] = []
  private lazy var queue: OperationQueue = OperationQueue()
  
  /// Default initializer for max pooling.
  /// - Parameter inputSize: Optional input size at this layer. If this is the first layer you will need to set this.
  public init(inputSize: TensorSize = TensorSize(array: [])) {
    self.inputSize = inputSize
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize,
         type
  }
  
  convenience public init(from decoder: Decoder) throws {
    self.init()
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
  }
  
  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
  }
  
  public func forward(tensor: Tensor) -> Tensor {
    func backwards(input: Tensor, gradient: Tensor) -> (Tensor, Tensor) {
      let deltas = gradient.value
      var poolingGradients: [[[Tensor.Scalar]]] = []
      
      queue.addSynchronousOperation { [weak self] in
        guard let sSelf = self else {
          return
        }
        
        guard let forwardPooledMaxIndicies = sSelf.poolingGradients.first(where: { $0.tensorId == input.id })?.indicies else {
          return
        }
        
        for i in 0..<deltas.count {
          let delta: [Tensor.Scalar] = deltas[i].flatten()
          var modifiableDeltas = delta
          
          var pooledGradients = [Tensor.Scalar].init(repeating: 0,
                                                     count: sSelf.inputSize.rows * sSelf.inputSize.columns).reshape(columns: sSelf.inputSize.columns)
              
          let indicies = forwardPooledMaxIndicies[i]
          
          indicies.forEach { index in
            pooledGradients[index.r][index.c] = modifiableDeltas.removeFirst()
          }
          
          poolingGradients.append(pooledGradients)
        }
      }
      
      if poolingGradients.isEmpty {
        return (gradient, Tensor())
      }
      
      return (Tensor(poolingGradients), Tensor())
    }
    
    var currentIndicies: [[PoolingIndex]] = []
    var results: [[[Tensor.Scalar]]] = []
    currentIndicies.reserveCapacity(tensor.value.count)

    tensor.value.forEach { input in
      let pool = pool(input: input)
      currentIndicies.append(pool.1)
      results.append(pool.0)
    }

    queue.addBarrierBlock {
      self.setGradients(indicies: currentIndicies, id: tensor.id)
    }
          
    let context = TensorContext(backpropagate: backwards)
    let out = Tensor(results, context: context)
    
    out.setGraph(tensor)
    
    return out
  }
  
  private func setGradients(indicies: [[PoolingIndex]], id: UUID) {
    self.poolingGradients.append(PoolingGradient(tensorId: id, indicies: indicies))
  }
  
  public func apply(gradients: Optimizer.Gradient, learningRate: Float) {
    poolingGradients.removeAll(keepingCapacity: true)
  }
  
  internal func pool(input: [[Tensor.Scalar]]) -> ([[Tensor.Scalar]], [PoolingIndex]) {
    var rowResults: [Tensor.Scalar] = []
    var results: [[Tensor.Scalar]] = []
    var pooledIndicies: [PoolingIndex] = []
        
    let rows = inputSize.rows
    let columns = inputSize.columns
            
    for r in stride(from: 0, through: rows, by: 2) {
      guard r < input.count else {
        continue
      }
      rowResults = []
      
      for c in stride(from: 0, through: columns, by: 2) {
        guard c < input[r].count else {
          continue
        }
        let current = input[r][c]
        let right = input[safe: r + 1]?[c] ?? 0
        let bottom = input[r][safe: c + 1] ?? 0
        let diag = input[safe: r + 1]?[safe: c + 1] ?? 0
        
        let indiciesToCheck = [(current, r,c),
                               (right ,r + 1, c),
                               (bottom, r, c + 1),
                               (diag, r + 1, c + 1)]
        
        let max = max(max(max(current, right), bottom), diag)
        if let firstIndicies = indiciesToCheck.first(where: { $0.0 == max }) {
          pooledIndicies.append(PoolingIndex(r: firstIndicies.1, c: firstIndicies.2))
        }
        rowResults.append(max)
      }
      
      results.append(rowResults)
    }
              
    return (results, pooledIndicies)
  }
}
