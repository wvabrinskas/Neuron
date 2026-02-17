//
//  EmptyRNNDataset.swift
//  Neuron
//
//  Created by William Vabrinskas on 2/16/26.
//

public class EmptyRNNDataset: VectorizableDataset<String> {
  
  /// Creates a placeholder dataset for inference-only RNN usage.
  ///
  /// - Parameter vocabSize: Vocabulary size expected by the loaded model.
  public init(vocabSize: Int) {
    super.init()
    self.vocabSize = vocabSize
  }
}
