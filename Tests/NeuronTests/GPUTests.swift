import XCTest
import NumSwift
@testable import Neuron


class GPUTests: XCTestCase {
  
  func testGPUActivationFunctionSize_multiple() {
    let inputSize = TensorSize(rows: 28, columns: 28, depth: 16)
    
    let inputTensor = Tensor.withRandom(size: inputSize, in: Float(-1)...Float(1))
    
    let gpu = GPUManager()
    let out = gpu.activate(inputTensor,
                           inputSize: inputSize,
                           activationType: .leakyRelu(limit: 0.1),
                           derivate: true)
    
    XCTAssertEqual(out.shape, [inputSize.rows, inputSize.columns, inputSize.depth])
  }
  
  
  func testGPUActivationFunctionSize() {
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
    
    let conv = Conv2d(filterCount: filterCount,
                      inputSize: inputSize,
                      padding: .same,
                      initializer: .heNormal)
    
    
    
    
  }
}
