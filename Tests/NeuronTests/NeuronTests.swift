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

  func test_tensor_Subscript() {
    let input: [[Tensor.Scalar]] = [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
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
    
    let inputTensor = Tensor(input)
    XCTAssertEqual(inputTensor[0..., ..<10, 0...].shape, [27, 10, 1])
  }
  
  func testTransConv2dLayer() {
    let inputShape = TensorSize(array: [10,10,1])
    
    let filterCount = 1
    
    let input: [[Tensor.Scalar]] = [0,0,1,0,0,0,0,1,0,0].as2D()
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
    
    let gradients: [[[Tensor.Scalar]]] = NumSwift.onesLike((outputShape[safe: 1, 0], outputShape[safe: 0, 0], filterCount))
    let backward = out.gradients(delta: Tensor(gradients), wrt: inputTensor)
    
    let expectedGradient: [[[Tensor.Scalar]]] = [[[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
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
    let r: [[[Tensor.Scalar]]] = [[[1.1,1.2,1.3],
                           [1.4,1.5,1.6]],
                          [[2.1,2.2,2.3],
                           [2.4,2.5,2.6]],
                          [[2.1,2.2,2.3],
                           [2.4,2.5,2.6]]]
    
    let testData: Tensor = Tensor(r)
    
    let layer = Flatten()
    layer.inputSize = TensorSize(array: r.shape)
    
    let out = layer.forward(tensor: testData)
    out.setGraph(testData)
    
    let rFlat: [Tensor.Scalar] = r.flatten()
    let backward = out.gradients(delta: Tensor(rFlat), wrt: testData)
    
    XCTAssert(backward.input.first?.value.shape == r.shape)
  }
  
  func testReshape() {
    let r: [Tensor.Scalar] = [1.1,1.2,1.3,
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
    
    let backward = out.gradients(delta: out.detached(), wrt: testData)
    
    XCTAssert(backward.input.first?.shape == testData.shape)
  }
  
  func testMaxPool() {
    let r: [[[Tensor.Scalar]]] = [[[0,1,0],
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
    
    let gradients: [[[Tensor.Scalar]]] = [[[1.0, 0.0],
                                   [0.0, 0.0]],
                                  [[1.0, 0.0],
                                   [0.0, 0.0]],
                                  [[1.0, 0.0],
                                   [0.0, 0.0]]]
    
    let backward = out.gradients(delta: Tensor(gradients), wrt: testData)
    
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
  
  func testConv2d_1x1_Filter() {
    let inputSize = TensorSize(array: [28,28,1])
    
    let filterCount = 1
    
    let input = Tensor.fillRandom(size: inputSize)
    
    let conv = Conv2d(filterCount: filterCount,
                      inputSize: inputSize,
                      strides: (2,2),
                      padding: .same,
                      filterSize: (1,1),
                      initializer: .heNormal,
                      biasEnabled: false)
    
    let out = conv.forward(tensor: input)
    
    let gradients = Tensor.fillRandom(size: conv.outputSize)
    let backward = out.gradients(delta: gradients, wrt: input)
    
    print(backward)
    
  }
  
  func testConv2d() {
    let inputSize = (10,10,1)
    
    let filterCount = 1
    let outputShape = [5,5,filterCount]
    
    let input: [[Tensor.Scalar]] = [0,0,1,0,0,0,0,1,0,0].as2D()
    
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
    
    let gradients: [[[Tensor.Scalar]]] = NumSwift.onesLike((out.shape[safe: 1, 0], out.shape[safe: 0, 0], filterCount))
    let backward = out.gradients(delta: Tensor(gradients), wrt: inputTensor)
    
    XCTAssert(backward.input.first?.shape == [inputSize.0,inputSize.1,inputSize.2])
  }
  
  func testUpsample7x7to28x28() {
    var random: [Tensor.Scalar] = []
    for _ in 0..<100 {
      random.append(Tensor.Scalar.random(in: 0...1))
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
    let adam = Adam(n, learningRate: 0.01, batchSize: 1)
    
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
    
    let adam = Adam(n, learningRate: 1, batchSize: 1)
    
    let input = Tensor([0.5,0.2,0.2,1.0])
    
    let out = adam([input]).first ?? Tensor()
    
    let expectedTensor = Tensor([[[0.95, 0.19, 0.95, 0.19, 0.95]]])
    
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
  
  func testLayerNorm_1d() {
    let input = Tensor([[1,0,1,0,1]])
    let norm = LayerNormalize(inputSize: input.shape.tensorSize)
    
    let out = norm.forward(tensor: input)
    out.setGraph(input)

    XCTAssert(out.isValueEqual(to: Tensor([[0.8164965, -1.2247449, 0.8164965, -1.2247449, 0.8164965]])))
    
    let delta = Tensor([[0.5, 0, 0.5, 0, 0.5]])
    
    let gradient = out.gradients(delta: delta, wrt: input)
    
    XCTAssert(gradient.input.first?.isEmpty == false)
    XCTAssert(gradient.weights.first![0..., 0..., 0..<input.shape.tensorSize.depth].isValueEqual(to: Tensor([0.40824825, 0.0, 0.40824825, 0.0, 0.40824825])))
    XCTAssert(gradient.input.first!.isValueEqual(to: Tensor([-2.3814485, 0.0, -2.3814485, 0.0, -2.3814485])))
  }
  
  func testLayerNorm_2d() {
    let input = Tensor([[1,0,1,0,1], [1,0,1,0,1]])
    let norm = LayerNormalize(inputSize: input.shape.tensorSize)
    
    let out = norm.forward(tensor: input)
    out.setGraph(input)

    XCTAssert(out.isValueEqual(to: Tensor([[0.8164965, -1.2247449, 0.8164965, -1.2247449, 0.8164965],
                                           [0.8164965, -1.2247449, 0.8164965, -1.2247449, 0.8164965]])))
    
    let delta = Tensor([[0.5, 0, 0.5, 0, 0.5],
                        [0.5, 0, 0.5, 0, 0.5]])
    
    let gradient = out.gradients(delta: delta, wrt: input)
    
    XCTAssert(gradient.input.first?.isEmpty == false)
    XCTAssert(gradient.weights.first![0..., 0..., 0..<input.shape.tensorSize.depth].isValueEqual(to: Tensor([[0.40824825, 0.0000, 0.40824825, 0.0000, 0.40824825],
                                                                                                             [0.40824825, 0.0000, 0.40824825, 0.0000, 0.40824825]])))
  }
  
  
  func testLayerNorm_3d() {
    let input = Tensor([[[1,0],
                         [1,0]],
                       [[1,0],
                        [1,0]]])
    let norm = LayerNormalize(inputSize: input.shape.tensorSize)
    
    let out = norm.forward(tensor: input)
    out.setGraph(input)

    XCTAssert(out.isValueEqual(to: Tensor([[[ 1.0000, -1.0000],
                                            [ 1.0000, -1.0000]],
                                           [[ 1.0000, -1.0000],
                                            [ 1.0000, -1.0000]]])))
    
    let delta = Tensor([[[0.5,0],
                         [0.5,0]],
                        [[0.5,0],
                         [0.5,0]]])
    
    let gradient = out.gradients(delta: delta, wrt: input)
    
    XCTAssert(gradient.input.first?.isEmpty == false)
    XCTAssert(gradient.weights.first![0..., 0..., 0..<input.shape.tensorSize.depth].isValueEqual(to: Tensor([[[0.5,0],
                                                                                                              [0.5,0]],
                                                                                                             [[0.5,0],
                                                                                                              [0.5,0]]])))
  }
  
  
  func testBatchNorm() {
    var batch: [Tensor] = []
    
    let batchSize = 10
    
    for i in 0..<batchSize {
      if i % 2 == 0 {
        batch.append(Tensor([0,1,0,1,0]))
      } else {
        batch.append(Tensor([1,0,1,0,1]))
      }
    }
    
    let norm = BatchNormalize(inputSize: TensorSize(array: [5,1,1]))
    
    norm.batchSize = batchSize
    norm.isTraining = true
    
    batch.concurrentBatchedForEach(workers: Constants.maxWorkers) { elements, workerIndex, indexRange, processingCount, workerId in
        let _ = norm.forward(tensorBatch: elements, context: .init(batchRange: indexRange,
                                                           batchProcessingCount: processingCount,
                                                           totalInBatch: batch.count,
                                                           threadId: workerId))
      
    }
    
    XCTAssertEqual(norm.welfordVariance.iterations, batchSize)
    XCTAssertEqual(norm.welfordVariance.m2s, [[[2.5,2.5,2.5,2.5,2.5]]]) // variance
    XCTAssertEqual(norm.welfordVariance.means, [[[0.5, 0.5, 0.5, 0.5, 0.5]]]) // mean
  }
  
  func testBatchNorm2d() {
    var batch: [Tensor] = []
    
    let batchSize = 10
    
    for i in 0..<batchSize {
      if i % 2 == 0 {
        batch.append(Tensor([0,1,0,1,0].as2D()))
      } else {
        batch.append(Tensor([1,0,1,0,1].as2D()))
      }
    }
    
    let norm = BatchNormalize(inputSize: TensorSize(array: [5,5,1]))
    
    norm.batchSize = batchSize
    norm.isTraining = true
    
    batch.concurrentBatchedForEach(workers: Constants.maxWorkers) { elements, workerIndex, indexRange, processingCount, workerId in
        let _ = norm.forward(tensorBatch: elements, context: .init(batchRange: indexRange,
                                                           batchProcessingCount: processingCount,
                                                           totalInBatch: batch.count,
                                                           threadId: workerId))
      
    }
    
    XCTAssertEqual(norm.welfordVariance.iterations, batchSize)
    XCTAssertEqual(norm.welfordVariance.m2s, [[2.5,2.5,2.5,2.5,2.5].as2D()]) // variance
    XCTAssertEqual(norm.welfordVariance.means, [[0.5, 0.5, 0.5, 0.5, 0.5].as2D()]) // mean
  }
  
  func testBatchNorm_isZero_withOneSample() {
    let inputSize = TensorSize(array: [3,1,1])
    let input = Tensor([1,2,3])
    let norm = BatchNormalize(inputSize: inputSize)
    
    norm.batchSize = 1
    norm.isTraining = true

    let out = norm.forward(tensorBatch: [input], context: .init())

    XCTAssertNotNil(out.first)
    
    XCTAssertEqual(out.first!.value.flatten(), [0,0,0])
  }
  
  func testBatchNorm3d() {
    var batch: [Tensor] = []
    
    let batchSize = 10
    
    for i in 0..<batchSize {
      if i % 2 == 0 {
        batch.append(Tensor([0,1,0,1,0].as3D()))
      } else {
        batch.append(Tensor([1,0,1,0,1].as3D()))
      }
    }
    
    let norm = BatchNormalize(inputSize: TensorSize(array: [5,5,5]))
    
    norm.batchSize = batchSize
    norm.isTraining = true
    
    batch.concurrentBatchedForEach(workers: Constants.maxWorkers) { elements, workerIndex, indexRange, processingCount, workerId in
        let _ = norm.forward(tensorBatch: elements, context: .init(batchRange: indexRange,
                                                           batchProcessingCount: processingCount,
                                                           totalInBatch: batch.count,
                                                           threadId: workerId))
      
    }
    
    XCTAssertEqual(norm.welfordVariance.iterations, batchSize)
    XCTAssertEqual(norm.welfordVariance.m2s, [2.5,2.5,2.5,2.5,2.5].as3D()) // variance
    XCTAssertEqual(norm.welfordVariance.means, [0.5, 0.5, 0.5, 0.5, 0.5].as3D()) // mean
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
    
    let gradient = out.gradients(delta: delta, wrt: input)
    
    XCTAssert(gradient.input.first?.isEmpty == false)
    XCTAssert(gradient.input.first!.isValueEqual(to:  Tensor([1, 0, 1, 0, 1].as3D())))
  }
  
}
