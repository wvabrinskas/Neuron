//
//  File.swift
//  
//
//  Created by William Vabrinskas on 3/18/24.
//
import Foundation
import NumSwift

/// Will decrease the size of the input tensor by half using a max pooling technique.
public final class AvgPool: BaseLayer {
  internal struct PoolingIndex: Hashable, Codable {
    var r: Int
    var c: Int
  }
  
  internal struct PoolingGradient: Hashable, Codable {
    static func == (lhs: AvgPool.PoolingGradient, rhs: AvgPool.PoolingGradient) -> Bool {
      lhs.tensorId == rhs.tensorId
    }
    
    var tensorId: UUID
    var indicies: [[PoolingIndex]]
  }
  
  internal var poolingGradients: [PoolingGradient] = []
  private lazy var queue: OperationQueue = OperationQueue()
  private let kernelSize: (Int, Int) = (rows: 2, columns: 2)
  
  /// Default initializer for max pooling.
  /// - Parameter inputSize: Optional input size at this layer. If this is the first layer you will need to set this.
  public init(inputSize: TensorSize = TensorSize(array: [])) {
    super.init(inputSize: inputSize,
               initializer: nil,
               biasEnabled: false,
               encodingType: .avgPool)
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
  
  public override func forward(tensor: Tensor) -> Tensor {
    func backwards(input: Tensor, gradient: Tensor) -> (Tensor, Tensor, Tensor) {
      let deltas = gradient.value
      var poolingGradients: [[[Tensor.Scalar]]] = []
      
      let rows = inputSize.rows
      let columns = inputSize.columns
                    
      for d in 0..<inputSize.depth {
        let inputValue = input.value[d]
        var gradientR = 0
        
        for r in stride(from: 0, through: rows, by: kernelSize.0) {
          guard r < rows else {
            continue
          }
          
          var gradientC = 0
          for c in stride(from: 0, through: columns, by: kernelSize.1) {
            guard c < columns else {
              continue
            }
            
            let current = inputValue[r][c]
            let right = inputValue[safe: r + 1]?[c] ?? 0
            let bottom = inputValue[r][safe: c + 1] ?? 0
            let diag = inputValue[safe: r + 1]?[safe: c + 1] ?? 0
            
            let indiciesToCheck = [current,
                                   right,
                                   bottom,
                                   diag]
            
            gradientC += 1
          }
          
          gradientR += 1
          
        }
        
      }

      if poolingGradients.isEmpty {
        return (gradient, Tensor(), Tensor())
      }
      
      return (Tensor(poolingGradients), Tensor(), Tensor())
    }
    
    var currentIndicies: [[PoolingIndex]] = []
    var results: [[[Tensor.Scalar]]] = []
    currentIndicies.reserveCapacity(inputSize.depth)

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
  
  override public func onInputSizeSet() {
    outputSize = TensorSize(array: [inputSize.columns / 2, inputSize.rows / 2, inputSize.depth])
  }
  
  private func setGradients(indicies: [[PoolingIndex]], id: UUID) {
    self.poolingGradients.append(PoolingGradient(tensorId: id, indicies: indicies))
  }
  
  public override func apply(gradients: Optimizer.Gradient, learningRate: Float) {
    poolingGradients.removeAll(keepingCapacity: true)
  }
  
  internal func pool(input: [[Tensor.Scalar]]) -> ([[Tensor.Scalar]], [PoolingIndex]) {
    var rowResults: [Tensor.Scalar] = []
    var results: [[Tensor.Scalar]] = []
    var pooledIndicies: [PoolingIndex] = []
        
    let rows = inputSize.rows
    let columns = inputSize.columns
            
    for r in stride(from: 0, through: rows, by: kernelSize.0) {
      guard r < rows else {
        continue
      }
      rowResults = []
      
      for c in stride(from: 0, through: columns, by: kernelSize.1) {
        guard c < columns else {
          continue
        }
        let current = input[r][c]
        let right = input[safe: r + 1]?[c] ?? 0
        let bottom = input[r][safe: c + 1] ?? 0
        let diag = input[safe: r + 1]?[safe: c + 1] ?? 0
        
        let indiciesToCheck = [current,
                               right,
                               bottom,
                               diag]
        
        let avg = indiciesToCheck.average
        rowResults.append(avg)
      }
      
      results.append(rowResults)
    }
              
    return (results, pooledIndicies)
  }
}
