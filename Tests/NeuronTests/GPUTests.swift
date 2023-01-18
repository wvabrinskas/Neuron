import XCTest
import NumSwift
@testable import Neuron


class GPUTests: XCTestCase {
  
  func testGPUActivationFunction() {
    let inputSize = TensorSize(rows: 1, columns: 10, depth: 1)
    
    let inputTensor = Tensor([-1, -1, -1, -1, 1, -1, -1, -1, -1, -1])
    
    let activation = ReLu()
    activation.inputSize = inputSize
    activation.device = GPU()
    
    let out = activation.forward(tensor: inputTensor)
    
    XCTAssertEqual(out.shape, [inputSize.columns, inputSize.rows, inputSize.depth])
    XCTAssertTrue(Tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).isValueEqual(to: out))
  }
  
  func testConv2dGPU() {
    let inputSize = TensorSize(rows: 28, columns: 28, depth: 16)

    let filterCount = 32
        
    let inputTensor = Tensor.withRandom(size: inputSize, in: Float(-1)...Float(1))
            
    let conv = Conv2d(filterCount: filterCount,
                      inputSize: inputSize,
                      padding: .same,
                      initializer: .heNormal)
    
    conv.device = GPU()
    
    let out = conv.forward(tensor: inputTensor)
    
    XCTAssertEqual(out.shape, [inputSize.columns, inputSize.rows, filterCount])
  }
}
