import XCTest
import NumSwift
@testable import Neuron

class GPUTests: XCTestCase {
  
  func testConv2d() {
    let gpuManager = GPUManager()
    
    let inputShape = (6,6,1)
    
    let filterCount = 2
    
    let input: [[Float]] = [1,1,1,1,1,1].as2D()
    
    let inputTensor = Tensor(input)
    
    let filters = [Tensor([[[0,1,0],
                            [0,1,0],
                            [0,1,0]]])]
    
    let outputShape = (3, 3, filters.count)
    
    let out = gpuManager.conv2d(inputTensor,
                                filters: filters,
                                biases: Tensor(1),
                                padding: .same,
                                filterSize: (3,3),
                                strides: (1,1),
                                inputSize: inputShape)
    
    //  let out = gpuManager.commit()
    print(out)
  }
  
  func testCPUConv2d() {
    let inputSize = (6,6,1)
    
    let filterCount = 1
    let outputShape = [12,12,filterCount]
    
    let input: [[Float]] = [1,1,1,1,1,1].as2D()
    
    let conv = Conv2d(filterCount: filterCount,
                      inputSize: [inputSize.0, inputSize.1, inputSize.2].tensorSize,
                      strides: (1,1),
                      padding: .same,
                      filterSize: (3,3),
                      initializer: .heNormal,
                      biasEnabled: false)
    
    conv.filters = [Tensor([[[0,1,0],
                             [0,1,0],
                             [0,1,0]]])]
    
    let inputTensor = Tensor(input)
    
    let out = conv.forward(tensor: inputTensor)
    print(out)
  }
}
