import Foundation

public final class ResNet: BaseTrainable {
  public override var name: String { get { "ResNet" } set {} }
  
  override public func compile() {
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
  
  override public func predict(_ data: Tensor, context: NetworkContext) -> Tensor {
    precondition(isCompiled, "Please call compile() on the \(self) before attempting to fit")
    
    var outputTensor = data
    
    layers.forEach { layer in
      let newTensor = layer.forward(tensor: outputTensor, context: context)
      if newTensor.graph == nil {
        newTensor.setGraph(outputTensor)
      }
      outputTensor = newTensor
    }
    
    return outputTensor
  }
  
}
