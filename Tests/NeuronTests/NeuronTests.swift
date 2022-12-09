import XCTest
import NumSwift
@testable import Neuron

extension XCTestCase {
  var isGithubCI: Bool {
    if let value = ProcessInfo.processInfo.environment["CI"] {
      return value == "true"
    }
    return false
  }
}

final class NeuronTests: XCTestCase {
  func testTransConv2dLayer() {
    let inputShape = TensorSize(array: [10,10,1])
    
    let filterCount = 1
    
    let input: [[Float]] = [0,0,1,0,0,0,0,1,0,0].as2D()
    let outputShape = [20, 20, filterCount]
    
    
    let conv = TransConv2d(filterCount: filterCount,
                           inputSize: inputShape,
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
    out.setGraph(inputTensor)
    
    XCTAssert(outputShape == out.shape)
    
    let gradients = NumSwift.onesLike((outputShape[safe: 1, 0], outputShape[safe: 0, 0], filterCount))
    let backward = out.gradients(delta: Tensor(gradients))
    
    let expectedGradient: [[[Float]]] = [[[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                                          [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                                          [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                                          [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                                          [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                                          [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                                          [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                                          [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                                          [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                                          [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]]]
    
    XCTAssert(backward.input.first!.isValueEqual(to: Tensor(expectedGradient)))
    XCTAssert(TensorSize(array: backward.input.first!.shape) == inputShape)
  }
  
  func testFlatten() {
    let r: [[[Float]]] = [[[1.1,1.2,1.3],
                           [1.4,1.5,1.6],
                           [1.4,1.5,1.6]],
                          [[2.1,2.2,2.3],
                           [2.4,2.5,2.6],
                           [2.4,2.5,2.6]],
                          [[2.1,2.2,2.3],
                           [2.4,2.5,2.6],
                           [2.4,2.5,2.6]]]
    
    let testData: Tensor = Tensor(r)
    
    let layer = Flatten()
    layer.inputSize = TensorSize(array: r.shape)
    
    let out = layer.forward(tensor: testData)
    out.setGraph(testData)
    
    let rFlat: [Float] = r.flatten()
    let backward = out.gradients(delta: Tensor(rFlat))
    
    XCTAssert(backward.input.first?.value.shape == r.shape)
  }
  
  func testReshape() {
    let r: [Float] = [1.1,1.2,1.3,
                      1.4,1.5,1.6,
                      1.4,1.5,1.6,
                      2.1,2.2,2.3,
                      2.4,2.5,2.6,
                      2.4,2.5,2.6,
                      2.1,2.2,2.3,
                      2.4,2.5,2.6,
                      2.4,2.5,2.6]
    
    let testData = Tensor(r)
    
    let size = TensorSize(array: [3,3,3])
    
    let layer = Reshape(to: size)
    layer.inputSize = r.shape.tensorSize
    
    let out = layer.forward(tensor: testData)
    out.setGraph(testData)
    
    XCTAssert(out.value.shape.tensorSize == size)
    
    let backward = out.gradients(delta: out.detached())
    
    XCTAssert(backward.input.first?.shape == testData.shape)
  }
  
  func testMaxPool() {
    let r: [[[Float]]] = [[[0,1,0],
                           [0,2,0],
                           [0,0,0]],
                          [[0,1,0],
                           [0,2,0],
                           [0,0,0]],
                          [[0,1,0],
                           [0,2,0],
                           [0,0,0]]]
    
    let testData = Tensor(r)
    
    let maxPool = MaxPool()
    maxPool.inputSize = TensorSize(array: [3,3,3])
    
    let data = testData
    
    let out = maxPool.forward(tensor: data)
    out.setGraph(data)
    
    let gradients: [[[Float]]] = [[[1.0, 0.0],
                                   [0.0, 0.0]],
                                  [[1.0, 0.0],
                                   [0.0, 0.0]],
                                  [[1.0, 0.0],
                                   [0.0, 0.0]]]
    
    let backward = out.gradients(delta: Tensor(gradients))
    
    let expected: Tensor = Tensor([[[0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0]],
                                   [[0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0]],
                                   [[0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0]]])
    
    XCTAssert(backward.input.first?.isValueEqual(to: expected) ?? false)
  }
  
  func testConv2d() {
    let inputSize = (10,10,1)
    
    let filterCount = 1
    let outputShape = [5,5,filterCount]
    
    let input: [[Float]] = [0,0,1,0,0,0,0,1,0,0].as2D()
    
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
    out.setGraph(inputTensor)

    XCTAssert(outputShape == out.value.shape)
    
    let gradients = NumSwift.onesLike((out.shape[safe: 1, 0], out.shape[safe: 0, 0], filterCount))
    let backward = out.gradients(delta: Tensor(gradients))
    
    XCTAssert(backward.input.first?.shape == [inputSize.0,inputSize.1,inputSize.2])
  }
  
  func testUpsample7x7to28x28() {
    var random: [Float] = []
    for _ in 0..<100 {
      random.append(Float.random(in: 0...1))
    }
    
    let n = Sequential {
      [
        Dense(7 * 7 * 1, inputs: 100, initializer: .heNormal),
        Reshape(to: [7,7,1].tensorSize),
        TransConv2d(filterCount: 1,
                    strides: (2,2),
                    padding: .same,
                    filterSize: (3,3),
                    initializer: .heNormal,
                    biasEnabled: false),
        ReLu(),
        TransConv2d(filterCount: 1,
                    strides: (2,2),
                    padding: .same,
                    filterSize: (3,3),
                    initializer: .heNormal,
                    biasEnabled: false),
        ReLu()
      ]
    }
    
    n.compile()
    
    let input = Tensor(random)
    let adam = Adam(n, learningRate: 0.01)
    
    let out = adam([input])
    
    XCTAssert(out.first?.shape == [28,28,1])
  }
  
  func testDense() {
    let dense = Dense(5, inputs: 4, biasEnabled: false)
    
    let n = Sequential {
      [
        dense,
      ]
    }
    
    n.compile()
    
    dense.weights = Tensor([[0.5, 0.5, 0.5, 0.5],
                            [0.1, 0.1, 0.1, 0.1],
                            [0.5, 0.5, 0.5, 0.5],
                            [0.1, 0.1, 0.1, 0.1],
                            [0.5, 0.5, 0.5, 0.5]])
    
    let adam = Adam(n, learningRate: 1)
    
    let input = Tensor([0.5,0.2,0.2,1.0])
    
    let out = adam([input]).first ?? Tensor()
    
    let expectedTensor = Tensor([[[0.95000005, 0.19, 0.95000005, 0.19, 0.95000005]]])
    
    XCTAssert(expectedTensor.isValueEqual(to: out))
  }
  
  func testGradientAccumulator() {
    let gradientsInput = [Tensor](repeating: Tensor(5), count: 5)
    let gradientsWeights = [Tensor](repeating: Tensor(10), count: 5)
    let gradientsBiases = [Tensor](repeating: Tensor(15), count: 5)
    let gradients = Tensor.Gradient(input: gradientsInput,
                                    weights: gradientsWeights,
                                    biases: gradientsBiases)
    
    let gradientsInput2 = [Tensor](repeating: Tensor(5), count: 5)
    let gradientsWeights2 = [Tensor](repeating: Tensor(20), count: 5)
    let gradientsBiases2 = [Tensor](repeating: Tensor(25), count: 5)
    let gradients2 = Tensor.Gradient(input: gradientsInput2,
                                     weights: gradientsWeights2,
                                     biases: gradientsBiases2)
    
    let accumulator = GradientAccumulator()
    accumulator.insert(gradients)
    accumulator.insert(gradients2)
    
    let result = accumulator.accumulate(clearAtEnd: true)
    
    result.weights.forEach { XCTAssert(Tensor(15).isValueEqual(to: $0)) }
    result.input.forEach { XCTAssert(Tensor(5).isValueEqual(to: $0)) }
    result.biases.forEach { XCTAssert(Tensor(20).isValueEqual(to: $0)) }
  }
  
  func testLayerNorm() {
    let input = Tensor([1,0,1,0,1])
    let norm = LayerNormalize(inputSize: [5,1,1].tensorSize)
    
    let out = norm.forward(tensor: input)
    out.setGraph(input)

    XCTAssert(out.isValueEqual(to: Tensor([0.8164965, -1.2247449, 0.8164965, -1.2247449, 0.8164965])))
    
    let delta = Tensor([0.5, 0, 0.5, 0, 0.5])
    
    let gradient = out.gradients(delta: delta)
    
    XCTAssert(gradient.input.first?.isEmpty == false)
    XCTAssert(gradient.input.first!.isValueEqual(to: Tensor([-1.4793792, -1.1920929e-07, -1.4793792, -1.1920929e-07, -1.4793792])))
  }
  
  func testBatchNorm() {
    let input = Tensor([1,0,1,0,1])
    let norm = BatchNormalize(inputSize: input.shape.tensorSize)
    
    let out = norm.forward(tensor: input)
    out.setGraph(input)

    XCTAssert(out.isValueEqual(to: Tensor([0.81647956, -1.2247194, 0.81647956, -1.2247194, 0.81647956])))
    
    let delta = Tensor([0.5, 0, 0.5, 0, 0.5])
    
    let gradient = out.gradients(delta: delta)
    
    XCTAssert(gradient.input.first?.isEmpty == false)
    XCTAssert(gradient.input.first!.isValueEqual(to: Tensor([-4.0823126, -0.00012750486, -4.0823126, -0.00012750486, -4.0823126])))
  }
  
  func testBatchNorm2d() {
    let input = Tensor([1,0,1,0,1].as2D())
    let norm = BatchNormalize(inputSize: input.shape.tensorSize)
    
    let out = norm.forward(tensor: input)
    out.setGraph(input)

    XCTAssert(out.isValueEqual(to: Tensor([0.81647956, -1.2247194, 0.81647956, -1.2247194, 0.81647956].as2D())))
    
    let delta = Tensor([0.5, 0, 0.5, 0, 0.5].as2D())
    
    let gradient = out.gradients(delta: delta)
    
    XCTAssert(gradient.input.first?.isEmpty == false)
    XCTAssert(gradient.input.first!.isValueEqual(to: Tensor([-4.082313, -0.00012769953, -4.082313, -0.00012769953, -4.082313].as2D())))
  }
  
  func testDropout() {
    let input = Tensor(NumSwift.onesLike((5,5,5)))
    
    let dropout = Dropout(0.5, inputSize: [5,5,5].tensorSize)
    
    let d: Tensor.Scalar = 1 / (1 - 0.5)
    let mask = Tensor([d,0,d,0,d].as3D())
    dropout.mask = mask
    
    let out = dropout.forward(tensor: input)
    out.setGraph(input)

    XCTAssert(out.isValueEqual(to: Tensor([2,0,2,0,2].as3D())))
    
    let delta = Tensor([0.5, 0.5, 0.5, 0.5, 0.5].as3D())
    
    let gradient = out.gradients(delta: delta)
    
    XCTAssert(gradient.input.first?.isEmpty == false)
    XCTAssert(gradient.input.first!.isValueEqual(to:  Tensor([1, 0, 1, 0, 1].as3D())))
    
    let dropoutNew = Dropout(0.5, inputSize: [5,5,5].tensorSize)
    let oldMask = dropoutNew.mask
    dropoutNew.apply(gradients: (Tensor(), Tensor()))
    
    XCTAssert(oldMask.isValueEqual(to: dropoutNew.mask) == false)
  }
  
}
