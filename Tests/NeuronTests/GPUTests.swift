import XCTest
import NumSwift
@testable import Neuron


class GPUTests: XCTestCase {
  
  func testGPUActivationFunctionSzie() {
    let inputSize = TensorSize(rows: 28, columns: 28, depth: 16)
    
    let inputTensor = Tensor.withRandom(size: inputSize, in: Float(-1)...Float(1))
    
    let activation = LeakyReLu(limit: 0.1)
    activation.inputSize = inputSize
    activation.device = GPU()
    
    let out = activation.forward(tensor: inputTensor)
    
    XCTAssertEqual(out.shape, [inputSize.rows, inputSize.columns, inputSize.depth])
  }
  
  func testConv2dGPUInNetwork() {
    let inputSize = TensorSize(rows: 28, columns: 28, depth: 16)

    let filterCount = 16
    let batchSize = 32
    let batchCount = 16
        
    let inputTensor = Tensor.withRandom(size: inputSize, in: Float(-1)...Float(1))
        
    let inputs = [Tensor].init(repeating: inputTensor, count: batchSize * batchCount)
    
    let n = Sequential {
      [
        Conv2d(filterCount: filterCount,
               inputSize: inputSize,
               padding: .same,
               initializer: .heNormal),
        BatchNormalize(),
        LeakyReLu(limit: 0.1),
        MaxPool(),
        Conv2d(filterCount: 32,
               padding: .same,
               initializer: .heNormal),
        BatchNormalize(),
        LeakyReLu(limit: 0.2),
        Dropout(0.5),
        MaxPool(),
        Flatten(),
        Dense(64, initializer: .heNormal),
        LeakyReLu(limit: 0.2),
        Dense(10, initializer: .heNormal),
        Softmax()
      ]
    }
    
    let optim = Adam(n,
                     device: GPU(),
                     learningRate: 0.001)
    
    let batched = inputs.batched(into: batchSize)
    
    for b in batched {
      let labels = [Tensor].init(repeating: Tensor([0,0,0,0,1,0,0,0,0,0]), count: b.count)
      let out = optim.fit(b, labels: labels, lossFunction: .crossEntropySoftmax)
      //print(out)
    }
  }
}
