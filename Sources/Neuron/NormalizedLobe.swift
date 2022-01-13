//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/13/22.
//

import Foundation
import UIKit

public class NormalizedLobe: Lobe {
  private let normalizer: BatchNormalizer = .init()
  
  public override func feed(inputs: [Float]) -> [Float] {
    let normalizedInputs = self.normalizer.normalize(activations: inputs)
    return super.feed(inputs: normalizedInputs)
  }
  
  public override func adjustWeights(_ constrain: ClosedRange<Float>? = nil) {
    for neuron in neurons {
      neuron.adjustWeights(constrain, normalizer: self.normalizer)
    }
  }
}
