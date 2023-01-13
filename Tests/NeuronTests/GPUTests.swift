import XCTest
import NumSwift
@testable import Neuron

class GPUTests: XCTestCase {
  
  func testConv2dMetal() {
    let inputShape = (6,6,1)
        
    let inputTensor = Tensor([[[Float]]].init(repeating: [[Float]].init(repeating: [Float].init(repeating: 1,
                                                                                                count: inputShape.0),
                                                                        count: inputShape.1),
                                              count: inputShape.2)  )
    
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
        }
        
      }
    })
  }
  
  func testConv2d() {
    
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
<<<<<<< HEAD
          print(out)
=======
          //print(out)
>>>>>>> 597064b (working on using MPSCNN)
        }
        
      }
    })
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
    
    conv.filters = [Tensor].init(repeating: Tensor([[[0,1,0],
                                                     [0,1,0],
                                                     [0,1,0]]]), count: filterCount)
    
    let inputTensor = Tensor(input)
    
    let out = conv.forward(tensor: inputTensor)
    print(out)
  
   
  }
}
