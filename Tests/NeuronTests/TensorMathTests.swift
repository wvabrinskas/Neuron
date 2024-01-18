//
//  File.swift
//  
//
//  Created by William Vabrinskas on 7/20/23.
//

import Foundation
import XCTest
import NumSwift
@testable import Neuron

final class TensorMathTests: XCTestCase {
  
  func test_split_dataIntegrity() {
    let tensor = Tensor([
                         [[1,1,1],
                          [2,2,2]],
                         
                         [[5,3,5],
                          [5,3,5]]
                        ])
    
    // axis 2
    let split = tensor.split(into: 1)
    XCTAssertEqual(split.shape, [2])
    XCTAssertTrue(split[0].isValueEqual(to: Tensor([[1,1,1],
                                                    [2,2,2]])))
    XCTAssertTrue(split[1].isValueEqual(to: Tensor([[5,3,5],
                                                    [5,3,5]])))
    // axis 0
    let split1 = tensor.split(into: 1, axis: 0)
    XCTAssertEqual(split1.shape, [2])
    XCTAssertTrue(split1[0].isValueEqual(to: Tensor([[[1,1,1]],
                                                     [[5,3,5]]])))
    XCTAssertTrue(split1[1].isValueEqual(to: Tensor([[[2,2,2]],
                                                     [[5,3,5]]])))
    
    let split2 = tensor.split(into: 1, axis: 1)
    XCTAssertEqual(split2.shape, [3])
    XCTAssertTrue(split2[0].isValueEqual(to: Tensor([[[1.0],[2.0]], [[5.0], [5.0]]])))
    XCTAssertTrue(split2[1].isValueEqual(to: Tensor([[[1.0],[2.0]], [[3.0], [3.0]]])))
    XCTAssertTrue(split2[2].isValueEqual(to: Tensor([[[1.0],[2.0]], [[5.0], [5.0]]])))
  }
  
  func test_split() {
    let tensor = Tensor(NumSwift.onesLike((10, 10, 10)))
    
    // axis 2
    let split = tensor.split(into: 2)
    XCTAssertEqual(split.shape, [5])
    split.forEach { XCTAssertEqual($0.shape, [10,10,2]) }
    
    // axis 0
    let split1 = tensor.split(into: 2, axis: 0)
    XCTAssertEqual(split1.shape, [5])
    split1.forEach { XCTAssertEqual($0.shape, [10,2,10]) }
    
    let split2 = tensor.split(into: 2, axis: 1)
    XCTAssertEqual(split2.shape, [5])
    split2.forEach { XCTAssertEqual($0.shape, [2,10,10]) }
  }
  
  func test_mean() {
    let tensor = Tensor([[[1,1,1],
                         [2,2,2]],
                         [[5,3,5],
                          [5,3,5]]])
    
    // axis -1
    let result = tensor.mean(axis: -1)
    XCTAssertTrue(Tensor([2.9166667]).isValueEqual(to: result))
    
    // axis 0
    let resultZero = tensor.mean(axis: 0)
    let expected = Tensor([[[1.5, 1.5, 1.5]],
                           [[5.0, 3.0, 5.0]]])
    XCTAssertTrue(expected.isValueEqual(to: resultZero))
    
    // axis 1
    let resultOne = tensor.mean(axis: 1)
    let expectedOne = Tensor([[[1.0], [2.0]],
                              [[4.3333335], [4.3333335]]])
    XCTAssertTrue(expectedOne.isValueEqual(to: resultOne))
    
    // axis 2
    let resultTwo = tensor.mean(axis: 2)
    let expectedTwo = Tensor([[3.0, 2.0, 3.0],
                              [3.5, 2.5, 3.5]])
    XCTAssertTrue(expectedTwo.isValueEqual(to: resultTwo))
  }
  
  func test_sum() {
    let tensor = Tensor([[[1,1,1],
                         [2,2,2]],
                         [[5,3,5],
                          [5,3,5]]])
    
    // axis -1
    let result = tensor.sum(axis: -1)
    XCTAssertTrue(Tensor([35.0]).isValueEqual(to: result))
    
    // axis 0
    let resultZero = tensor.sum(axis: 0)
    let expected = Tensor([[[3.0, 3.0, 3.0]],
                           [[10.0, 6.0, 10.0]]])
    XCTAssertTrue(expected.isValueEqual(to: resultZero))
    
    // axis 1
    let resultOne = tensor.sum(axis: 1)
    let expectedOne = Tensor([[[3.0], [6.0]],
                              [[13.0], [13.0]]])
    XCTAssertTrue(expectedOne.isValueEqual(to: resultOne))
    
    // axis 2
    let resultTwo = tensor.sum(axis: 2)
    let expectedTwo = Tensor([[6.0, 4.0, 6.0],
                              [7.0, 5.0, 7.0]])
    XCTAssertTrue(expectedTwo.isValueEqual(to: resultTwo))
  }
  
  func test_subtract() {
    let tensor = Tensor([[[1,1,1],
                         [2,2,2]],
                         [[5,3,5],
                          [5,3,5]]])
    
    // axis -1
    let result = tensor.subtract(axis: -1)
    XCTAssertTrue(Tensor([-35.0]).isValueEqual(to: result))
    
    // axis 0
    let resultZero = tensor.subtract(axis: 0)
    let expected = Tensor([[[-1.0, -1.0, -1.0]],
                           [[0.0, 0.0, 0.0]]])
    XCTAssertTrue(expected.isValueEqual(to: resultZero))
    
    // axis 1
    let resultOne = tensor.subtract(axis: 1)
    let expectedOne = Tensor([[[-1.0], [-2.0]],
                              [[-3.0], [-3.0]]])
    XCTAssertTrue(expectedOne.isValueEqual(to: resultOne))
    
    // axis 2
    let resultTwo = tensor.subtract(axis: 2)
    let expectedTwo = Tensor([[-4.0, -2.0, -4.0],
                              [-3.0, -1.0, -3.0]])
    XCTAssertTrue(expectedTwo.isValueEqual(to: resultTwo))
  }
  
  func test_multiply() {
    let tensor = Tensor([[[1,1,1],
                         [2,2,2]],
                         [[5,3,5],
                          [5,3,5]]])
    
    // axis -1
    let result = tensor.multiply(axis: -1)
    XCTAssertTrue(Tensor([45000.0]).isValueEqual(to: result))
    
    // axis 0
    let resultZero = tensor.multiply(axis: 0)
    let expected = Tensor([[[2.0, 2.0, 2.0]],
                           [[25.0, 9.0, 25.0]]])
    XCTAssertTrue(expected.isValueEqual(to: resultZero))
    
    // axis 1
    let resultOne = tensor.multiply(axis: 1)
    let expectedOne = Tensor([[[1.0], [8.0]],
                              [[75.0], [75.0]]])
    XCTAssertTrue(expectedOne.isValueEqual(to: resultOne))
    
    // axis 2
    let resultTwo = tensor.multiply(axis: 2)
    let expectedTwo = Tensor([[5.0, 3.0, 5.0],
                              [10.0, 6.0, 10.0]])
    XCTAssertTrue(expectedTwo.isValueEqual(to: resultTwo))
  }
  
  func test_norm() {
    let tensor = Tensor([[[1,1,1],
                         [2,2,2]],
                         [[5,3,5],
                          [5,3,5]]])
    
    // axis -1
    let result = tensor.norm(axis: -1)
    XCTAssertTrue(Tensor([11.532562]).isValueEqual(to: result))
    
    // axis 0
    let resultZero = tensor.norm(axis: 0)
    let expected = Tensor([[[2.236068, 2.236068, 2.236068]],
                           [[7.071068, 4.2426405, 7.071068]]])
    XCTAssertTrue(expected.isValueEqual(to: resultZero))
    
    // axis 1
    let resultOne = tensor.norm(axis: 1)
    let expectedOne = Tensor([[[1.7320508], [3.4641016]],
                              [[7.6811457], [7.6811457]]])
    XCTAssertTrue(expectedOne.isValueEqual(to: resultOne))
    
    // axis 2
    let resultTwo = tensor.norm(axis: 2)
    let expectedTwo = Tensor([[5.0990195, 3.1622777, 5.0990195],
                              [5.3851647, 3.6055512, 5.3851647]])
    XCTAssertTrue(expectedTwo.isValueEqual(to: resultTwo))
  }
  
  func test_sumOfSquares() {
    let tensor = Tensor([[[1,1,1],
                         [2,2,2]],
                         [[5,3,5],
                          [5,3,5]]])
    
    // axis -1
    let result = tensor.sumOfSquares(axis: -1)
    XCTAssertTrue(Tensor([133.0]).isValueEqual(to: result))
    
    // axis 0
    let resultZero = tensor.sumOfSquares(axis: 0)
    let expected = Tensor([[[5.0, 5.0, 5.0]],
                           [[50.0, 18.0, 50.0]]])
    XCTAssertTrue(expected.isValueEqual(to: resultZero))
    
    // axis 1
    let resultOne = tensor.sumOfSquares(axis: 1)
    let expectedOne = Tensor([[[3.0], [12.0]],
                              [[59.0], [59.0]]])
    XCTAssertTrue(expectedOne.isValueEqual(to: resultOne))
    
    // axis 2
    let resultTwo = tensor.sumOfSquares(axis: 2)
    let expectedTwo = Tensor([[26.0, 10.0, 26.0],
                              [29.0, 13.0, 29.0]])
    XCTAssertTrue(expectedTwo.isValueEqual(to: resultTwo))
  }
}
