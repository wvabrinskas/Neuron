//
//  EmptyRNNDataset.swift
//  Neuron
//
//  Created by William Vabrinskas on 2/16/26.
//

public class EmptyRNNDataset: VectorizableDataset<String> {
  
  public init(vocabSize: Int) {
    super.init()
    self.vocabSize = vocabSize
  }
}
