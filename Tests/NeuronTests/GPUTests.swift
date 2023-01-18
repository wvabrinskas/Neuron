import XCTest
import NumSwift
@testable import Neuron


class GPUTests: XCTestCase {
  
  func testGPUActivationFunctionSize() {
    let inputSize = TensorSize(rows: 1, columns: 28, depth: 1)
    
    let inputTensor = Tensor.withRandom(size: inputSize, in: Float(-1)...Float(1))
    
    let activation = LeakyReLu(limit: 0.1)
    activation.inputSize = inputSize
    activation.device = GPU()
    
    let out = activation.forward(tensor: inputTensor)
    
    XCTAssertEqual(out.shape, [inputSize.columns, inputSize.rows, inputSize.depth])
  }
  
  func testAsyncTensor() {
    let tensor = AsyncTensor()
    
    let inputShape = (6,6,6)
    
    let filterCount = 256
    
    let inputTensor = Tensor([Float].init(repeating: 1, count: inputShape.0).as3D())
    
    let singleFilter = [[[Float]]].init(repeating: [[Float]].init(repeating: [0,1,0],
                                                                  count: 3),
                                        count: inputShape.2)
        
    let filters = [Tensor].init(repeating: Tensor(singleFilter), count: filterCount)
    
    let inputs = [Tensor].init(repeating: inputTensor, count: 32)
    
    let device = GPU()
    inputs.concurrentForEach(workers: 16, { input, index in

      for f in 0..<filterCount {
        for i in 0..<inputTensor.value.count {
          
          let currentFilter = filters[f].value[i]
          let currentInput = inputTensor.value[i]
          
          let out = device.conv2d(signal: currentInput,
                                  filter: currentFilter,
                                  strides: (1,1),
                                  padding: .same,
                                  filterSize: (3,3),
                                  inputSize: (inputShape.0, inputShape.1))
          print(out)
        }
        
      }
    })
  }
  
  func testConv2dGPUInNetwork() {
    let inputSize = TensorSize(rows: 28, columns: 28, depth: 16)

    let filterCount = 16
    let batchSize = 32
    let batchCount = 16
        
    let inputTensor = Tensor.withRandom(size: inputSize, in: Float(-1)...Float(1))
        
    let inputs = [Tensor].init(repeating: inputTensor, count: batchSize * batchCount)
    
    let conv = Conv2d(filterCount: filterCount,
                      inputSize: inputSize,
                      padding: .same,
                      initializer: .heNormal)
    
    
    
    
  }
}
