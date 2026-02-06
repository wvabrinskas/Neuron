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
      let rows = self.inputSize.rows
      let columns = self.inputSize.columns
      let kRows = self.kernelSize.rows
      let kCols = self.kernelSize.columns
      let kernelArea = Tensor.Scalar(kRows * kCols)
      let gradCols = gradient._size.columns
      
      var outStorage = ContiguousArray<Tensor.Scalar>(repeating: 0, count: rows * columns * self.inputSize.depth)
      
      for d in 0..<self.inputSize.depth {
        let gradSlice = gradient.depthSlice(d)
        let depthOffset = d * rows * columns
        var gradientR = 0
        
        for r in 0..<rows {
          var gradientC = 0
          for c in stride(from: 0, to: columns, by: kCols) {
            let avgPoolGradient = gradSlice[gradientR * gradCols + gradientC]
            let delta = avgPoolGradient / kernelArea
            
            for kc in 0..<kCols {
              if c + kc < columns {
                outStorage[depthOffset + r * columns + c + kc] = delta
              }
            }
            
            gradientC += 1
          }
          
          if (r + 1) % kRows == 0 {
            gradientR += 1
          }
        }
      }
      
      return (Tensor(outStorage, size: self.inputSize), Tensor(), Tensor())
    }
    
    let outStorage = poolFlat(input: tensor)
    let outRows = inputSize.rows / kernelSize.rows
    let outCols = inputSize.columns / kernelSize.columns
    let outSize = TensorSize(rows: outRows, columns: outCols, depth: inputSize.depth)

    let context = TensorContext(backpropagate: backwards)
    let out = Tensor(outStorage, size: outSize, context: context)
    
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
  
  internal func poolFlat(input: Tensor) -> ContiguousArray<Tensor.Scalar> {
    let rows = inputSize.rows
    let columns = inputSize.columns
    let kRows = kernelSize.rows
    let kCols = kernelSize.columns
    let kernelArea = Tensor.Scalar(kRows * kCols)
    let outRows = rows / kRows
    let outCols = columns / kCols
    
    var results = ContiguousArray<Tensor.Scalar>(repeating: 0, count: outRows * outCols * inputSize.depth)
    
    for d in 0..<inputSize.depth {
      let slice = input.depthSlice(d)
      let depthOffset = d * outRows * outCols
      var outIdx = 0
      
      for r in stride(from: 0, to: rows, by: kRows) {
        for c in stride(from: 0, to: columns, by: kCols) {
          var sum: Tensor.Scalar = 0
          for kr in 0..<kRows {
            for kc in 0..<kCols {
              let sr = r + kr
              let sc = c + kc
              if sr < rows && sc < columns {
                sum += slice[sr * columns + sc]
              }
            }
          }
          results[depthOffset + outIdx] = sum / kernelArea
          outIdx += 1
        }
      }
    }
    
    return results
  }
}
