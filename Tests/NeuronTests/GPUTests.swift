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
  
  func testConv2dGPU_Texture() {
    let inputSize = TensorSize(rows: 16, columns: 16, depth: 8)
    
    let inputTensor = Tensor([[[Float]]].init(repeating: [[Float]].init(repeating: [Float].init(repeating: 1,
                                                                                                count: inputSize.columns),
                                                                        count: inputSize.rows),
                                              count: inputSize.depth))
    
    let filterSize = (3, 3, inputSize.depth)
    let filter = Tensor([[[Float]]].init(repeating: [[Float]].init(repeating: [0,1,0],
                                                                   count: filterSize.0),
                                         count: filterSize.2))
    
    let filters = [Tensor].init(repeating: filter, count: 64)
    
    let manager = GPUManager()
    let padding: NumSwift.ConvPadding = .same
    let filterSizeMap = (filterSize.0, filterSize.1)
    let strides = (1,1)
    
    let out = manager.conv2d(inputTensor,
                             filters: filters,
                             padding: padding,
                             filterSize: filterSizeMap,
                             strides: strides,
                             inputSize: inputSize)
    
    let cpuOut = Conv2d(filterCount: filters.count,
                        inputSize: inputSize,
                        strides: strides,
                        padding: padding,
                        filterSize: filterSizeMap,
                        initializer: .heNormal,
                        biasEnabled: false)
    
    cpuOut.filters = filters
    
    let cpuOutVal = cpuOut.forward(tensor: inputTensor)
    
    XCTAssert(out.isValueEqual(to: cpuOutVal))
    
  }
}
