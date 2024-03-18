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
      var poolingGradients: [[[Tensor.Scalar]]] = []
      
      let rows = inputSize.rows
      let columns = inputSize.columns
                    
      for d in 0..<inputSize.depth {
        var gradientR = 0
        
        var results: [[Float]] = []
        for r in 0..<rows {
          var rowResult: [Float] = []

          var gradientC = 0
          for c in stride(from: 0, through: columns, by: kernelSize.1) {
            guard c < columns else {
              continue
            }
            
            let avgPoolGradient = gradient.value[d][gradientR][gradientC]
            let delta = avgPoolGradient / (Tensor.Scalar(kernelSize.0) * Tensor.Scalar(kernelSize.1))
          
            for _ in 0..<kernelSize.1 {
              rowResult.append(delta)
            }
            
            gradientC += 1
          }
          
          if (r + 1) % kernelSize.0 == 0 {
            gradientR += 1
          }
        
          results.append(rowResult)
        }
        
        poolingGradients.append(results)
      }

      if poolingGradients.isEmpty {
        return (gradient, Tensor(), Tensor())
      }
      
      return (Tensor(poolingGradients), Tensor(), Tensor())
    }
    
    var results: [[[Tensor.Scalar]]] = []

    tensor.value.forEach { input in
      let pool = pool(input: input)
      results.append(pool)
    }

    let context = TensorContext(backpropagate: backwards)
    let out = Tensor(results, context: context)
    
    out.setGraph(tensor)
    
    return out
  }
  
  override public func onInputSizeSet() {
    outputSize = TensorSize(array: [inputSize.columns / 2, inputSize.rows / 2, inputSize.depth])
  }

  
  public override func apply(gradients: Optimizer.Gradient, learningRate: Float) {
    //
  }
  
  internal func pool(input: [[Tensor.Scalar]]) -> [[Tensor.Scalar]] {
    var rowResults: [Tensor.Scalar] = []
    var results: [[Tensor.Scalar]] = []
        
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
              
    return results
  }
}
