import XCTest
import NumSwift
@testable import Neuron

class GPUTests: XCTestCase {
  private let gpuManager = GPUManager.shared
  
  func testConv2d() {
    let inputShape = (6,6,1)
    
    let filterCount = 1
    
    let input: [[Float]] = [1,1,1,1,1,1].as2D()
    let outputShape = (3, 3, filterCount)
    
    let inputTensor = Tensor(input)
    
    let filters = [Tensor([[[0,1,0],
                            [0,1,0],
                            [0,1,0]]])]
    
    gpuManager.conv2d(inputTensor,
                      filters: filters,
                      biases: Tensor(1),
                      filterCount: filterCount,
                      filterSize: (3,3),
                      strides: (2,2),
                      inputSize: inputShape,
                      outputSize: outputShape)
    
  }
  
  func testCPUConv2d() {
    let inputSize = (6,6,1)
    
    let filterCount = 1
    let outputShape = [12,12,filterCount]
    
    let input: [[Float]] = [1,1,1,1,1,1].as2D()
    
    let conv = Conv2d(filterCount: filterCount,
                      inputSize: [inputSize.0, inputSize.1, inputSize.2].tensorSize,
                      strides: (2,2),
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
