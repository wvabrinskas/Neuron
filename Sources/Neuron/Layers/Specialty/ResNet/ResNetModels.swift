//
//  ResNetModels.swift
//  Neuron
//
//  Created by William Vabrinskas on 12/1/24.
//

public final class ResConvolutionalBlockConfiguration: Codable {
  public let filters: Int
  public let strides: (rows: Int, columns: Int)
  public let inputSize: TensorSize
  public let filterSize: (rows: Int, columns: Int)
  
  enum CodingKeys: String, CodingKey {
    case filterSize,
         strides,
         filters,
         inputSize
  }

  required convenience public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    let filterSize = try container.decodeIfPresent([Int].self, forKey: .filterSize) ?? [3,3]
    let filterCount = try container.decodeIfPresent(Int.self, forKey: .filters) ?? 1
    let strides = try container.decodeIfPresent([Int].self, forKey: .strides) ?? [1,1]
    
    let filterSizeTuple = (filterSize[safe: 1, 3], filterSize[safe: 0, 3])
    let stridesTuple = (strides[safe: 1, 3], strides[safe: 0, 3])
    
    self.init(filters: filterCount,
              strides: stridesTuple,
              inputSize: inputSize,
              filterSize: filterSizeTuple)
    
  }
  
  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode([filterSize.rows, filterSize.columns], forKey: .filterSize)
    try container.encode([strides.rows, strides.columns], forKey: .strides)
    try container.encode(filters, forKey: .filters)
  }
  
  public init(filters: Int,
              strides: (rows: Int, columns: Int),
              inputSize: TensorSize,
              filterSize: (rows: Int, columns: Int) = (3,3)) {
    self.filters = filters
    self.strides = strides
    self.inputSize = inputSize
    self.filterSize = filterSize
  }
}
