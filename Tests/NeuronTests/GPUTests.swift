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
  
  func testConv2dGPU_Texture() {
    let inputSize = TensorSize(rows: 6, columns: 6, depth: 6)
        
    let inputTensor = Tensor([[[Float]]].init(repeating: [[Float]].init(repeating: [Float].init(repeating: 1,
                                                                                                count: inputSize.columns),
                                                                        count: inputSize.rows),
                                              count: inputSize.depth))
    
    let filterSize = (3, 3, inputSize.depth)
    let filter = Tensor([[[Float]]].init(repeating: [[Float]].init(repeating: [0,1,0],
                                                                   count: filterSize.0),
                                         count: filterSize.2))
    
    let manager = GPUManager()
    
    let out = manager.conv2dTexture(inputTensor,
                                    filter: filter,
                                    padding: .same,
                                    filterSize: (filterSize.0, filterSize.1),
                                    strides: (1,1),
                                    inputSize: inputSize)
    
    let cpuOut = Conv2d(filterCount: 1,
                        inputSize: inputSize,
                        strides: (1,1),
                        padding: .same,
                        filterSize: (filterSize.0, filterSize.1),
                        initializer: .heNormal,
                        biasEnabled: false)
    
    cpuOut.filters = [filter]
    
    let cpuOutVal = cpuOut.forward(tensor: inputTensor)
    
    print(cpuOutVal)

   // XCTAssert(out.isValueEqual(to: cpuOutVal))
    
    print(out)
    
  }
}
