//
//  VectorizableDataset.swift
//  Neuron
//
//  Created by William Vabrinskas on 1/9/26.
//

@testable import Neuron

open class VectorizableDataset<VectorItem: Hashable>: RNNSupportedDataset {

  public let vectorizer: Vectorizer<VectorItem> = .init()
  
  public var vocabSize: Int = 0
  
  public func oneHot(_ items: [VectorItem]) -> Tensor {
    vectorizer.oneHot(items)
  }
  
  public func vectorize(_ items: [VectorItem]) -> Tensor {
    Tensor(items.map { [[Tensor.Scalar(vectorizer.vector[$0, default: 0])]] })
  }
  
  public func getWord(for data: Tensor, oneHot: Bool) -> [VectorItem] {
    if oneHot == false {
      let intArray = data.value.map { $0.map { $0.map { Int($0) }}}
      if let int = intArray[safe: 0]?[safe: 0] {
        return vectorizer.unvectorize(int)
      }
      
      return []
    } else {
      return vectorizer.unvectorizeOneHot(data)
    }
  }
  
  public func build() async -> Neuron.RNNSupportedDatasetData {
    ([], [])
  }
}
