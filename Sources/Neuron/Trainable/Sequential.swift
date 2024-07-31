//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation

public final class Sequential<N: TensorNumeric>: BaseTrainable<N> {
  public override var name: String { get { "Sequential" } set {}}

  public override func predict(_ data: Tensor) -> Tensor {
    precondition(isCompiled, "Please call compile() on the \(self) before attempting to fit")
    
    var outputTensor = data
    
    layers.forEach { layer in
      let newTensor = layer.forward(tensor: outputTensor)
      if newTensor.graph == nil {
        newTensor.setGraph(outputTensor)
      }
      outputTensor = newTensor
    }
    
    return outputTensor
  }
  
  public override func compile() {
    var inputSize: TensorSize = TensorSize(array: [])
    var i = 0
    layers.forEach { layer in
      if i == 0 && layer.inputSize.isEmpty {
        fatalError("The first layer should contain an input size")
      }
      
      if i > 0 {
        layer.inputSize = inputSize
      }
      
      inputSize = layer.outputSize
      i += 1
    }
    
    isCompiled = true
  }
  
  public override func exportWeights() throws -> [[Tensor]] {
    guard isCompiled else {
      throw LayerErrors.generic(error: "Please compile the trainable first before attempting to export weights.")
    }
    
    return try layers.map { try $0.exportWeights() }
  }
  
  public override func importWeights(_ weights: [[Tensor]]) throws {
    guard isCompiled else {
      throw LayerErrors.generic(error: "Please compile the trainable first before attempting to import weights.")
    }
    
    for i in 0..<layers.count {
      let layer = layers[i]
      let weights = weights[i]
      try layer.importWeights(weights)
    }
  }
}
