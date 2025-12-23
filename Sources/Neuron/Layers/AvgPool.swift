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

  private var kernelSize: TensorSize
  /// Default initializer for max pooling.
  /// - Parameter inputSize: Optional input size at this layer. If this is the first layer you will need to set this.
  public init(inputSize: TensorSize? = nil, kernelSize: (rows: Int, columns: Int) = (rows: 2, columns: 2)) {
    self.kernelSize = .init(rows: kernelSize.rows, columns: kernelSize.columns, depth: 1)
    
    super.init(inputSize: inputSize,
               biasEnabled: false,
               encodingType: .avgPool)
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize,
         kernelSize,
         type
  }
  
  convenience public required init(from decoder: Decoder) throws {
    self.init()
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.kernelSize = try container.decodeIfPresent(TensorSize.self, forKey: .kernelSize) ?? TensorSize(rows: 2, columns: 2, depth: 1)
  }
  
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
    try container.encode(kernelSize, forKey: .kernelSize)
  }
  
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    func backwards(input: Tensor, gradient: Tensor, wrt: Tensor?) -> (Tensor, Tensor, Tensor) {
      var poolingGradients: [[[Tensor.Scalar]]] = []
      
      let rows = inputSize.rows
      let columns = inputSize.columns
                    
      for d in 0..<inputSize.depth {
        var gradientR = 0
        
        var results: [[Tensor.Scalar]] = []
        for r in 0..<rows {
          var rowResult: [Tensor.Scalar] = []

          var gradientC = 0
          for c in stride(from: 0, through: columns, by: kernelSize.columns) {
            guard c < columns else {
              continue
            }
            
            let avgPoolGradient = gradient.value[d][gradientR][gradientC]
            let delta = avgPoolGradient / (Tensor.Scalar(kernelSize.rows) * Tensor.Scalar(kernelSize.columns))
          
            for _ in 0..<kernelSize.columns {
              rowResult.append(delta)
            }
            
            gradientC += 1
          }
          
          if (r + 1) % kernelSize.rows == 0 {
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
    
    let results: [[[Tensor.Scalar]]] = pool(input: tensor)

    let context = TensorContext(backpropagate: backwards)
    let out = Tensor(results, context: context)
    
    out.setGraph(tensor)
    
    return out
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = TensorSize(array: [inputSize.columns / kernelSize.columns, inputSize.rows / kernelSize.rows, inputSize.depth])
  }

  
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
    //
  }
  
  internal func pool(input: Tensor) -> [[[Tensor.Scalar]]] {
    var results: [[[Tensor.Scalar]]] = []
        
    let rows = inputSize.rows
    let columns = inputSize.columns
            
    for d in 0..<inputSize.depth {

      var colResults: [[Tensor.Scalar]] = []

      for r in stride(from: 0, through: rows, by: kernelSize.rows) {
        guard r < rows else {
          continue
        }
        
        var rowResults: [Tensor.Scalar] = []

        for c in stride(from: 0, through: columns, by: kernelSize.columns) {
          guard c < columns else {
            continue
          }
          
          let average = input[c..<c+kernelSize.columns, r..<r+kernelSize.rows, d..<d+1].mean().asScalar()
          
          rowResults.append(average)
        }
        
        colResults.append(rowResults)
      }
      
      results.append(colResults)
    }
    
    return results
  }
}
