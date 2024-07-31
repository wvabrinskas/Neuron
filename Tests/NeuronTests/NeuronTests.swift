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
  typealias N = Float

  func test_tensor_Subscript() {
    let input: [[Tensor<N>.Scalar]] = [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                              0,  0,  0,  0,  0,  0,  0,  0,  1],
                            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                              0,  0,  0,  0,  0,  0,  1,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                              0,  0,  0,  0,  1,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                              0,  0,  0,  0,  1,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                              0,  0,  0,  0,  0,  0,  0,  0,  1],
                            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                              0,  0,  0,  0,  0,  0,  0,  0,  1],
                            [ 0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                              0,  0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                              0,  0,  0,  0,  0,  0,  1,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,
                              0,  0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                              1,  0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                              0,  0,  0,  0,  0,  0,  1,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                              0,  0,  0,  0,  0,  1,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                              0,  0,  0,  0,  1,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                              0,  0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                              0,  0,  0,  0,  0,  1,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                              0,  0,  0,  0,  0,  0,  0,  0,  1],
                            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                              0,  1,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,
                              0,  0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,
                              0,  0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                              0,  0,  0,  0,  0,  1,  0,  0,  0]]
    
    let inputTensor = Tensor<N>(input)
    XCTAssertEqual(inputTensor[0..., ..<10, 0...].shape, [27, 10, 1])
  }
  
  func testTransConv2dLayer() {
    let inputShape = TensorSize(array: [10,10,1])
    
    let filterCount = 1
    
    let input: [[Tensor<N>.Scalar]] = [0,0,1,0,0,0,0,1,0,0].as2D()
    let outputShape = [20, 20, filterCount]
    
    
    let conv = TransConv2d<Float>(filterCount: filterCount,
                           inputSize: inputShape,
                           strides: (2,2),
                           padding: .same,
                           filterSize: (3,3),
                           initializer: .heNormal,
                           biasEnabled: false)
    
    conv.filters = [Tensor<N>([[[0,1,0],
                             [0,1,0],
                             [0,1,0]]])]
    
    let inputTensor = Tensor<N>(input)
    
    let out = conv.forward(tensor: inputTensor)
    out.setGraph(inputTensor)
    
    XCTAssert(outputShape == out.shape)
    
    let gradients: [[[Tensor<N>.Scalar]]] = NumSwift.onesLike((outputShape[safe: 1, 0], outputShape[safe: 0, 0], filterCount))
    let backward = out.gradients(delta: Tensor<N>(gradients))
    
    let expectedGradient: [[[Tensor<N>.Scalar]]] = [[[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                                          [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                                          [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                                          [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                                          [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                                          [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                                          [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                                          [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                                          [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                                          [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]]]
    
    XCTAssert(backward.input.first!.isValueEqual(to: Tensor<N>(expectedGradient)))
    XCTAssert(TensorSize(array: backward.input.first!.shape) == inputShape)
  }
  
  func testFlatten() {
    let r: [[[Tensor<N>.Scalar]]] = [[[1.1,1.2,1.3],
                           [1.4,1.5,1.6]],
                          [[2.1,2.2,2.3],
                           [2.4,2.5,2.6]],
                          [[2.1,2.2,2.3],
                           [2.4,2.5,2.6]]]
    
    let testData: Tensor<N> = Tensor<N>(r)
    
    let layer = Flatten<Float>()
    layer.inputSize = TensorSize(array: r.shape)
    
    let out = layer.forward(tensor: testData)
    out.setGraph(testData)
    
    let rFlat: [Tensor<N>.Scalar] = r.flatten()
    let backward = out.gradients(delta: Tensor<N>(rFlat))
    
    XCTAssert(backward.input.first?.value.shape == r.shape)
  }
  
  func testReshape() {
    let r: [Tensor<N>.Scalar] = [1.1,1.2,1.3,
                      1.4,1.5,1.6,
                      1.4,1.5,1.6,
                      2.1,2.2,2.3,
                      2.4,2.5,2.6,
                      2.4,2.5,2.6,
                      2.1,2.2,2.3,
                      2.4,2.5,2.6,
                      2.4,2.5,2.6]
    
    let testData = Tensor<N>(r)
    
    let size = TensorSize(array: [3,3,3])
    
    let layer = Reshape<Float>(to: size)
    layer.inputSize = r.shape.tensorSize
    
    let out = layer.forward(tensor: testData)
    out.setGraph(testData)
    
    XCTAssert(out.value.shape.tensorSize == size)
    
    let backward = out.gradients(delta: out.detached())
    
    XCTAssert(backward.input.first?.shape == testData.shape)
  }
  
  func testMaxPool() {
    let r: [[[Tensor<N>.Scalar]]] = [[[0,1,0],
                           [0,2,0],
                           [0,0,0]],
                          [[0,1,0],
                           [0,2,0],
                           [0,0,0]],
                          [[0,1,0],
                           [0,2,0],
                           [0,0,0]]]
    
    let testData = Tensor<N>(r)
    
    let maxPool = MaxPool<Float>()
    maxPool.inputSize = TensorSize(array: [3,3,3])
    
    let data = testData
    
    let out = maxPool.forward(tensor: data)
    out.setGraph(data)
    
    let gradients: [[[Tensor<N>.Scalar]]] = [[[1.0, 0.0],
                                   [0.0, 0.0]],
                                  [[1.0, 0.0],
                                   [0.0, 0.0]],
                                  [[1.0, 0.0],
                                   [0.0, 0.0]]]
    
    let backward = out.gradients(delta: Tensor<N>(gradients))
    
    let expected: Tensor<N> = Tensor<N>([[[0.0, 0.0, 0.0],
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
    
    let input: [[Tensor<N>.Scalar]] = [0,0,1,0,0,0,0,1,0,0].as2D()
    
    let conv = Conv2d<Float>(filterCount: filterCount,
                      inputSize: [inputSize.0, inputSize.1, inputSize.2].tensorSize,
                      strides: (2,2),
                      padding: .same,
                      filterSize: (3,3),
                      initializer: .heNormal,
                      biasEnabled: false)
    
    conv.filters = [Tensor<N>([[[0,1,0],
                             [0,1,0],
                             [0,1,0]]])]
    
    let inputTensor = Tensor<N>(input)
    
    let out = conv.forward(tensor: inputTensor)
    out.setGraph(inputTensor)

    XCTAssert(outputShape == out.value.shape)
    
    let gradients: [[[Tensor<N>.Scalar]]] = NumSwift.onesLike((out.shape[safe: 1, 0], out.shape[safe: 0, 0], filterCount))
    let backward = out.gradients(delta: Tensor<N>(gradients))
    
    XCTAssert(backward.input.first?.shape == [inputSize.0,inputSize.1,inputSize.2])
  }
  
  func testUpsample7x7to28x28() {
    var random: [Tensor<N>.Scalar] = []
    for _ in 0..<100 {
      random.append(Tensor<N>.Scalar.random(in: 0...1))
    }
    
    let n = Sequential<Float> {
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
    
    let input = Tensor<N>(random)
    let adam = Adam<Float>(n, learningRate: 0.01)
    
    let out = adam([input])
    
    XCTAssert(out.first?.shape == [28,28,1])
  }
  
  func testDense() {
    let dense = Dense<Float>(5, inputs: 4, biasEnabled: false)
    
    let n = Sequential<Float> {
      [
        dense,
      ]
    }
    
    n.compile()
    
    dense.weights = Tensor<N>([[0.5, 0.5, 0.5, 0.5],
                            [0.1, 0.1, 0.1, 0.1],
                            [0.5, 0.5, 0.5, 0.5],
                            [0.1, 0.1, 0.1, 0.1],
                            [0.5, 0.5, 0.5, 0.5]])
    
    let adam = Adam<Float>(n, learningRate: 1)
    
    let input = Tensor<N>([0.5,0.2,0.2,1.0])
    
    let out = adam([input]).first ?? Tensor<N>()
    
    let expectedTensor = Tensor<N>([[[0.95, 0.19, 0.95, 0.19, 0.95]]])
    
    XCTAssert(expectedTensor.isValueEqual(to: out))
  }
  
  func testGradientAccumulator() {
    let gradientsInput = [Tensor<N>](repeating: Tensor<N>(5), count: 5)
    let gradientsWeights = [Tensor<N>](repeating: Tensor<N>(10), count: 5)
    let gradientsBiases = [Tensor<N>](repeating: Tensor<N>(15), count: 5)
    let gradients = Tensor<N>.Gradient(input: gradientsInput,
                                    weights: gradientsWeights,
                                    biases: gradientsBiases)
    
    let gradientsInput2 = [Tensor<N>](repeating: Tensor<N>(5), count: 5)
    let gradientsWeights2 = [Tensor<N>](repeating: Tensor<N>(20), count: 5)
    let gradientsBiases2 = [Tensor<N>](repeating: Tensor<N>(25), count: 5)
    let gradients2 = Tensor<N>.Gradient(input: gradientsInput2,
                                     weights: gradientsWeights2,
                                     biases: gradientsBiases2)
    
    let accumulator = GradientAccumulator()
    accumulator.insert(gradients)
    accumulator.insert(gradients2)
    
    let result = accumulator.accumulate(clearAtEnd: true)
    
    result.weights.forEach { XCTAssert(Tensor<N>(15).isValueEqual(to: $0)) }
    result.input.forEach { XCTAssert(Tensor<N>(5).isValueEqual(to: $0)) }
    result.biases.forEach { XCTAssert(Tensor<N>(20).isValueEqual(to: $0)) }
  }
  
  func testLayerNorm() {
    let input = Tensor<N>([1,0,1,0,1])
    let norm = LayerNormalize<Float>(inputSize: [5,1,1].tensorSize)
    
    let out = norm.forward(tensor: input)
    out.setGraph(input)

    XCTAssert(out.isValueEqual(to: Tensor<N>([0.8164965, -1.2247449, 0.8164965, -1.2247449, 0.8164965])))
    
    let delta = Tensor<N>([0.5, 0, 0.5, 0, 0.5])
    
    let gradient = out.gradients(delta: delta)
    
    XCTAssert(gradient.input.first?.isEmpty == false)
    XCTAssert(gradient.input.first!.isValueEqual(to: Tensor<N>([-1.4793792, -1.1920929e-07, -1.4793792, -1.1920929e-07, -1.4793792])))
  }
  
  func testBatchNorm() {
    let input = Tensor<N>([1,0,1,0,1])
    let norm = BatchNormalize<Float>(inputSize: input.shape.tensorSize)
    
    let out = norm.forward(tensor: input)
    out.setGraph(input)

    XCTAssert(out.isValueEqual(to: Tensor<N>([0.81647956, -1.2247194, 0.81647956, -1.2247194, 0.81647956])))
    
    let delta = Tensor<N>([0.5, 0, 0.5, 0, 0.5])
    
    let gradient = out.gradients(delta: delta)
    
    XCTAssert(gradient.input.first?.isEmpty == false)
    XCTAssert(gradient.input.first!.isValueEqual(to: Tensor<N>([-4.0823126, -0.00012750486, -4.0823126, -0.00012750486, -4.0823126])))
  }
  
  func testBatchNorm2d() {
    let input = Tensor<N>([1,0,1,0,1].as2D())
    let norm = BatchNormalize<Float>(inputSize: input.shape.tensorSize)
    
    let out = norm.forward(tensor: input)
    out.setGraph(input)

    XCTAssert(out.isValueEqual(to: Tensor<N>([0.81647956, -1.2247194, 0.81647956, -1.2247194, 0.81647956].as2D())))
    
    let delta = Tensor<N>([0.5, 0, 0.5, 0, 0.5].as2D())
    
    let gradient = out.gradients(delta: delta)
    
    XCTAssert(gradient.input.first?.isEmpty == false)
    XCTAssert(gradient.input.first!.isValueEqual(to: Tensor<N>([-4.082313, -0.00012769953, -4.082313, -0.00012769953, -4.082313].as2D())))
  }
  
  func testDropout() {
    let input = Tensor<N>(NumSwift.onesLike((5,5,5)))
    
    let dropout = Dropout<Float>(0.5, inputSize: [5,5,5].tensorSize)
    
    let d: Tensor<N>.Scalar = 1 / (1 - 0.5)
    let mask = Tensor<N>([d,0,d,0,d].as3D())
    dropout.mask = mask
    
    let out = dropout.forward(tensor: input)
    out.setGraph(input)

    XCTAssert(out.isValueEqual(to: Tensor<N>([2,0,2,0,2].as3D())))
    
    let delta = Tensor<N>([0.5, 0.5, 0.5, 0.5, 0.5].as3D())
    
    let gradient = out.gradients(delta: delta)
    
    XCTAssert(gradient.input.first?.isEmpty == false)
    XCTAssert(gradient.input.first!.isValueEqual(to:  Tensor<N>([1, 0, 1, 0, 1].as3D())))
    
    let dropoutNew = Dropout<Float>(0.5, inputSize: [5,5,5].tensorSize)
    let oldMask = dropoutNew.mask
    dropoutNew.apply(gradients: (Tensor<N>(), Tensor<N>()), learningRate: 0.05)
    
    XCTAssert(oldMask.isValueEqual(to: dropoutNew.mask) == false)
  }
  
}
