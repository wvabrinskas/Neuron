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
  public init(inputSize: TensorSize? = nil,
              kernelSize: (rows: Int, columns: Int) = (rows: 2, columns: 2),
              linkId: String = UUID().uuidString) {
    self.kernelSize = .init(rows: kernelSize.rows, columns: kernelSize.columns, depth: 1)
    
    super.init(inputSize: inputSize,
               biasEnabled: false,
               linkId: linkId,
               encodingType: .avgPool)
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize,
         kernelSize,
         type,
         linkId
  }
  
  /// Decodes an AvgPool layer from a serialized model.
  ///
  /// - Parameter decoder: Decoder used during model loading.
  /// - Throws: An error if required values cannot be decoded.
  convenience public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let linkId = try container.decodeIfPresent(String.self, forKey: .linkId) ?? UUID().uuidString
    self.init(linkId: linkId)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.kernelSize = try container.decodeIfPresent(TensorSize.self, forKey: .kernelSize) ?? TensorSize(rows: 2, columns: 2, depth: 1)
  }
  
  /// Encodes average-pooling configuration.
  ///
  /// - Parameter encoder: Encoder used for serialization.
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
    try container.encode(kernelSize, forKey: .kernelSize)
    try container.encode(linkId, forKey: .linkId)
  }
  
  /// Performs average pooling over each kernel window.
  ///
  /// - Parameters:
  ///   - tensor: Input tensor.
  ///   - context: Network execution context.
  /// - Returns: Pooled tensor with average-distribution backpropagation context.
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    func backwards(input: Tensor, gradient: Tensor, wrt: Tensor?) -> (Tensor, Tensor, Tensor) {
      let rows = self.inputSize.rows
      let columns = self.inputSize.columns
      let kRows = self.kernelSize.rows
      let kCols = self.kernelSize.columns
      let kernelArea = Tensor.Scalar(kRows * kCols)
      let gradCols = gradient.size.columns
      let inSliceSize = rows * columns
      let outStorage = TensorStorage.create(count: inSliceSize * self.inputSize.depth)

      for d in 0..<self.inputSize.depth {
        let gradPtr  = gradient.storage.pointer + d * gradient.size.rows * gradCols
        let outPtr   = outStorage.pointer + d * inSliceSize
        var gradientR = 0

        for r in 0..<rows {
          var gradientC = 0
          for c in stride(from: 0, to: columns, by: kCols) {
            let delta = gradPtr[gradientR * gradCols + gradientC] / kernelArea
            for kc in 0..<kCols {
              if c + kc < columns {
                outPtr[r * columns + c + kc] = delta
              }
            }
            gradientC += 1
          }
          if (r + 1) % kRows == 0 {
            gradientR += 1
          }
        }
      }

      return (Tensor(storage: outStorage, size: self.inputSize), Tensor(), Tensor())
    }

    let outRows = inputSize.rows / kernelSize.rows
    let outCols = inputSize.columns / kernelSize.columns
    let outStorage = poolFlat(input: tensor)
    let outSize = TensorSize(rows: outRows, columns: outCols, depth: inputSize.depth)

    let tensorContext = TensorContext(backpropagate: backwards)
    let out = Tensor(storage: outStorage, size: outSize, context: tensorContext)

    out.setGraph(tensor)

    return super.forward(tensor: out, context: context)
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = TensorSize(array: [inputSize.columns / kernelSize.columns, inputSize.rows / kernelSize.rows, inputSize.depth])
  }

  
  /// AvgPool has no trainable parameters, so this is a no-op.
  ///
  /// - Parameters:
  ///   - gradients: Ignored.
  ///   - learningRate: Ignored.
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
    //
  }
  
  internal func poolFlat(input: Tensor) -> TensorStorage {
    let rows = inputSize.rows
    let columns = inputSize.columns
    let kRows = kernelSize.rows
    let kCols = kernelSize.columns
    let kernelArea = Tensor.Scalar(kRows * kCols)
    let outRows = rows / kRows
    let outCols = columns / kCols
    let outSliceSize = outRows * outCols
    let inSliceSize  = rows * columns

    let results = TensorStorage.create(count: outSliceSize * inputSize.depth)

    for d in 0..<inputSize.depth {
      let inPtr  = input.storage.pointer + d * inSliceSize
      let outPtr = results.pointer + d * outSliceSize
      var outIdx = 0

      for r in stride(from: 0, to: rows, by: kRows) {
        for c in stride(from: 0, to: columns, by: kCols) {
          var sum: Tensor.Scalar = 0
          for kr in 0..<kRows {
            for kc in 0..<kCols {
              let sr = r + kr
              let sc = c + kc
              if sr < rows && sc < columns {
                sum += inPtr[sr * columns + sc]
              }
            }
          }
          outPtr[outIdx] = sum / kernelArea
          outIdx += 1
        }
      }
    }

    return results
  }
}
