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

  
  func test_addition_and_Relu() {
    let size = TensorSize(array: [5,5,1])
    let inputL = Tensor.fillWith(value: 1.0, size: size)
    inputL.label = "inputL"
    let inputR = Tensor.fillWith(value: 0.5, size: size)
    inputR.label = "inputR"
    
    let relu = ReLu(inputSize: size)
    
    let add = inputL + inputR
    
    let out = relu.forward(tensor: add, context: .init())
        
    print(out)
    
    let delta = Tensor.fillRandom(size: size)
    
    let gradsWRTL = out.gradients(delta: delta, wrt: inputL)
    let gradsWRTR = out.gradients(delta: delta, wrt: inputR)
    let grads = out.gradients(delta: delta)
    
    XCTAssertEqual(gradsWRTL.input.count, 2)
    XCTAssertEqual(gradsWRTR.input.count, 2)
    // 3 because addition adds both inputs (lhs and rhs) to the graph in a single array.
    XCTAssertEqual(grads.input.count, 3)
  }
  
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
    
    XCTAssert(backward.input.first?.shape == r.shape)
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
    
    XCTAssert(out.shape.tensorSize == size)
    
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
    
    let filterCount = 12
    
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
    
    XCTAssertEqual(conv.outputSize.asArray, [14,14,filterCount])

    let backward = out.gradients(delta: gradients, wrt: input)
    
    XCTAssertEqual(backward.weights.first?.shape, [1,1,filterCount])

    // assert that it doesnt crash
    conv.apply(gradients: (backward.weights[0], backward.biases[0]), learningRate: 0.001)
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

    XCTAssert(outputShape == out.shape)
    
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
    
    XCTAssert(expectedTensor.isValueEqual(to: out, accuracy: 0.00001))
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
  
  func testInstanceNorm_1d() {
    let input = Tensor([[1,0,1,0,1]])
    let norm = InstanceNormalize(inputSize: input.shape.tensorSize)
    
    let out = norm.forward(tensor: input)
    out.setGraph(input)

    XCTAssert(out.isValueEqual(to: Tensor([[0.8164965, -1.2247449, 0.8164965, -1.2247449, 0.8164965]]), accuracy: 0.00001))
    
    let delta = Tensor([[0.5, 0, 0.5, 0, 0.5]])
    
    let gradient = out.gradients(delta: delta, wrt: input)
    
    XCTAssert(gradient.input.first?.isEmpty == false)
    XCTAssert(gradient.weights.first!.isValueEqual(to: Tensor([1.2247448, 1.5], size: .init(array: [1,1,2])), accuracy: 0.00001))
    XCTAssert(gradient.input.first!.isValueEqual(to: Tensor([7.3000486e-08, 0.0, 7.3000486e-08, 0.0, 7.3000486e-08]), accuracy: 0.00001))
  }
  
  func testInstanceNorm_2d() {
    let input = Tensor([[1,0,1,0,1], [1,0,1,0,1]])
    let norm = InstanceNormalize(inputSize: input.shape.tensorSize)
    
    let out = norm.forward(tensor: input)
    out.setGraph(input)

    XCTAssert(out.isValueEqual(to: Tensor([[0.8164965, -1.2247449, 0.8164965, -1.2247449, 0.8164965],
                                           [0.8164965, -1.2247449, 0.8164965, -1.2247449, 0.8164965]]), accuracy: 0.00001))
    
    let delta = Tensor([[0.5, 0, 0.5, 0, 0.5],
                        [0.5, 0, 0.5, 0, 0.5]])
    
    let gradient = out.gradients(delta: delta, wrt: input)
    
    XCTAssert(gradient.input.first?.isEmpty == false)
    XCTAssert(gradient.weights.first!.isValueEqual(to: Tensor([2.4494896, 3.0], size: .init(array: [1,1,2])), accuracy: 0.00001))
    XCTAssert(gradient.input.first!.isValueEqual(to: Tensor([[7.3000486e-08, 0.0, 7.3000486e-08, 0.0, 7.3000486e-08],
                                                             [7.3000486e-08, 0.0, 7.3000486e-08, 0.0, 7.3000486e-08]]), accuracy: 0.00001))
  }
  
  
  func testInstanceNorm_3d() {
    let input = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    let norm = InstanceNormalize(inputSize: input.shape.tensorSize)
    
    let out = norm.forward(tensor: input)
    out.setGraph(input)

    XCTAssert(out.isValueEqual(to: Tensor([[[ -1.3416407, -0.4472136],
                                            [ 0.4472136, 1.3416407]],
                                           [[ -1.3416407, -0.4472136],
                                            [ 0.4472136, 1.3416407]]]), accuracy: 0.00001))
    
    let delta = out.onesLike()
    
    let gradient = out.gradients(delta: delta, wrt: input)
    
    XCTAssert(gradient.input.first?.isEmpty == false)
    XCTAssert(gradient.input.first!.isValueEqual(to: .fillWith(value: 0, size: input.size), accuracy: 0.0001))
    XCTAssert(gradient.weights.first!.isValueEqual(to: .init([0, 0, 4, 4], size: .init(rows: 1, columns: 2, depth: 2)), accuracy: 0.0001))
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
    // m2s and means are now [ContiguousArray<Scalar>] (one flat slice per depth)
    norm.welfordVariance.m2s.forEach { slice in
      slice.forEach { scalar in XCTAssertEqual(scalar, 2.5, accuracy: 0.00001) }
    }
    norm.welfordVariance.means.forEach { slice in
      slice.forEach { scalar in XCTAssertEqual(scalar, 0.5, accuracy: 0.00001) }
    }
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
    
    norm.welfordVariance.m2s.forEach { slice in
      slice.forEach { scalar in
        XCTAssertEqual(scalar, 2.5, accuracy: 0.001)
      }
    }
    
    norm.welfordVariance.means.forEach { slice in
      slice.forEach { scalar in
        XCTAssertEqual(scalar, 0.5, accuracy: 0.001)
      }
    }
  }
  
  func testBatchNorm_isZero_withOneSample() {
    let inputSize = TensorSize(array: [3,1,1])
    let input = Tensor([1,2,3])
    let norm = BatchNormalize(inputSize: inputSize)
    
    norm.batchSize = 1
    norm.isTraining = true

    let out = norm.forward(tensorBatch: [input], context: .init())

    XCTAssertNotNil(out.first)
    
    XCTAssertEqual(out.first!.storage, [0,0,0])
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
    norm.welfordVariance.m2s.forEach { slice in
      slice.forEach { scalar in XCTAssertEqual(scalar, 2.5, accuracy: 0.001) }
    }
    norm.welfordVariance.means.forEach { slice in
      slice.forEach { scalar in XCTAssertEqual(scalar, 0.5, accuracy: 0.001) }
    }
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
  
  // MARK: - Flat Storage Tests
  
  func testFlatStorageInit_direct() {
    let storage = Tensor.Value([1, 2, 3, 4, 5, 6])
    let size = TensorSize(rows: 2, columns: 3, depth: 1)
    let tensor = Tensor(storage, size: size)
    
    XCTAssertEqual(tensor.shape, [3, 2, 1])
    XCTAssertEqual(tensor.storage.count, 6)
    XCTAssertEqual(tensor.size, size)
  }
  
  func testFlatStorageInit_roundTrip3D() {
    let data: Tensor.Data = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
    let tensor = Tensor(data)
    
    // Verify shape: cols=3, rows=2, depth=2
    XCTAssertEqual(tensor.shape, [3, 2, 2])
    XCTAssertEqual(tensor.storage.count, 12)
    
    // Verify round-trip through value property
    let roundTripped = tensor.storage.reshape(columns: 3).batched(into: 2)
    XCTAssertEqual(roundTripped, data)
  }
  
  func testFlatStorageInit_roundTrip2D() {
    let data: [[Tensor.Scalar]] = [[1, 2, 3], [4, 5, 6]]
    let tensor = Tensor(data)
    
    // 2D stored as depth=1: cols=3, rows=2, depth=1
    XCTAssertEqual(tensor.shape, [3, 2, 1])
    XCTAssertEqual(tensor.storage.count, 6)
    
    // value property returns 3D
    let roundTripped = tensor.storage.reshape(columns: 3)
    XCTAssertEqual([roundTripped], [data])
  }
  
  func testFlatStorageInit_roundTrip1D() {
    let data: [Tensor.Scalar] = [1, 2, 3, 4, 5]
    let tensor = Tensor(data)
    
    // 1D stored as depth=1, rows=1: cols=5, rows=1, depth=1
    XCTAssertEqual(tensor.shape, [5, 1, 1])
    XCTAssertEqual(tensor.storage.count, 5)
    
    // value property returns 3D: [[[1, 2, 3, 4, 5]]]
    let roundTripped = tensor.storage.reshape(columns: 5).batched(into: 1)
    XCTAssertEqual(roundTripped, [[data]])
  }
  
  func testFlatStorageInit_scalar() {
    let tensor = Tensor(Tensor.Scalar(42))
    
    XCTAssertEqual(tensor.shape, [1, 1, 1])
    XCTAssertEqual(tensor.storage.count, 1)
    XCTAssertEqual(tensor.asScalar(), 42)
    XCTAssertEqual(tensor.storage, [42])
  }
  
  func testFlatStorageInit_empty() {
    let tensor = Tensor()
    
    XCTAssertEqual(tensor.shape, [0, 0, 0])
    XCTAssertTrue(tensor.isEmpty)
    XCTAssertEqual(tensor.storage.count, 0)
    XCTAssertEqual(tensor.storage, [])
  }
  
  func testFlatSubscript_getSet() {
    let data: Tensor.Data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    let tensor = Tensor(data)
    
    // Test get: tensor[col, row, depth]
    XCTAssertEqual(tensor[0, 0, 0], 1)
    XCTAssertEqual(tensor[1, 0, 0], 2)
    XCTAssertEqual(tensor[0, 1, 0], 3)
    XCTAssertEqual(tensor[1, 1, 0], 4)
    XCTAssertEqual(tensor[0, 0, 1], 5)
    XCTAssertEqual(tensor[1, 1, 1], 8)
    
    // Test set
    tensor.storage[tensor.flatIndex(column: 0, row: 0, depth: 1)] = 99
    XCTAssertEqual(tensor[0, 0, 1], 99)
  }
  
  func testFlatStorage_debugDescription() {
    let data: Tensor.Data = [[[1, 2], [3, 4]]]
    let tensor = Tensor(data)
    
    let description = tensor.debugDescription
    XCTAssertTrue(description.contains("shape: (col: 2, rows: 2, depth: 1)"))
    XCTAssertTrue(description.contains("[1.0, 2.0]"))
    XCTAssertTrue(description.contains("[3.0, 4.0]"))
  }
  
  func testFlatStorage_codableRoundTrip() {
    let data: Tensor.Data = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
    let original = Tensor(data)
    
    do {
      let encoded = try JSONEncoder().encode(original)
      let decoded = try JSONDecoder().decode(Tensor.self, from: encoded)
      
      XCTAssertTrue(decoded.isValueEqual(to: original))
      XCTAssertEqual(decoded.shape, original.shape)
      XCTAssertEqual(decoded.storage.count, original.storage.count)
    } catch {
      XCTFail("Codable round-trip failed: \(error)")
    }
  }
  
  func testFlatStorage_detached() {
    let data: Tensor.Data = [[[1, 2], [3, 4]]]
    let tensor = Tensor(data)
    let detached = tensor.detached()
    
    XCTAssertTrue(detached.isValueEqual(to: tensor))
    XCTAssertEqual(detached.id, tensor.id)
    XCTAssertEqual(detached.shape, tensor.shape)
    XCTAssertEqual(detached.storage, tensor.storage)
  }
  
  func testFlatStorage_copy() {
    let data: Tensor.Data = [[[1, 2], [3, 4]]]
    let tensor = Tensor(data)
    let copied = tensor.copy()
    
    XCTAssertTrue(copied.isValueEqual(to: tensor))
    // copy creates a new ID
    XCTAssertNotEqual(copied.id, tensor.id)
  }
  
  func testFlatStorage_clip() {
    let data: Tensor.Data = [[[0.5, -0.5, 2.0, -3.0]]]
    let tensor = Tensor(data)
    tensor.clip(1.0)
    
    XCTAssertEqual(tensor[0, 0, 0], 0.5)
    XCTAssertEqual(tensor[1, 0, 0], -0.5)
    XCTAssertEqual(tensor[2, 0, 0], 1.0)
    XCTAssertEqual(tensor[3, 0, 0], -1.0)
  }
  
  func testFlatStorage_isValueEqual_accuracy() {
    let t1 = Tensor([[[1.0, 2.0]]])
    let t2 = Tensor([[[1.00001, 2.00001]]])
    
    XCTAssertTrue(t1.isValueEqual(to: t2, accuracy: 0.001))
    XCTAssertFalse(t1.isValueEqual(to: t2, accuracy: 0.000001))
  }
  
  func testFlatStorage_fillRandom() {
    let size = TensorSize(rows: 3, columns: 4, depth: 2)
    let tensor = Tensor.fillRandom(size: size)
    
    XCTAssertEqual(tensor.shape, [4, 3, 2])
    XCTAssertEqual(tensor.storage.count, 24)
    
    // Verify values are in [0,1]
    for val in tensor.storage {
      XCTAssertTrue(val >= 0 && val <= 1)
    }
  }
  
  func testFlatStorage_fillWith() {
    let size = TensorSize(rows: 2, columns: 3, depth: 2)
    let tensor = Tensor.fillWith(value: 7.0, size: size)
    
    XCTAssertEqual(tensor.shape, [3, 2, 2])
    for val in tensor.storage {
      XCTAssertEqual(val, 7.0)
    }
  }
  
  func testFlatStorage_raggedArrayNormalized() {
    // Ragged 3D array: different row sizes in different depth slices
    let ragged: Tensor.Data = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]]]
    let tensor = Tensor(ragged)
    
    // Should normalize to max dimensions: cols=3, rows=2, depth=2
    XCTAssertEqual(tensor.shape, [3, 2, 2])
    XCTAssertEqual(tensor.storage.count, 12)
    
    // First depth slice: [1,2,3], [4,5,6]
    XCTAssertEqual(tensor[0, 0, 0], 1)
    XCTAssertEqual(tensor[2, 1, 0], 6)
    
    // Second depth slice: [7,8,9], then zero-padded row
    XCTAssertEqual(tensor[0, 0, 1], 7)
    XCTAssertEqual(tensor[2, 0, 1], 9)
    XCTAssertEqual(tensor[0, 1, 1], 0) // zero-padded
  }
  
}
