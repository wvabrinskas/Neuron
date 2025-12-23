//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/1,9/23.
//

import Foundation
import XCTest
import NumSwift
@testable import Neuron

final class LayerTests: XCTestCase {
  
  func testGlobalAveragePool() {
    let inputSize: TensorSize = .init(rows: 16, columns: 16, depth: 16)
    
    let layer = GlobalAvgPool(inputSize: inputSize)
    
    XCTAssertEqual(layer.outputSize, .init(rows: 1, columns: inputSize.depth, depth: 1))
    
    let input = Tensor.fillWith(value: 0.5, size: inputSize)
    
    let out = layer.forward(tensor: input, context: .init())
    
    XCTAssertEqual(TensorSize(array: out.shape).columns, inputSize.depth)
    
    let error = Tensor.fillWith(value: 0.1, size: layer.outputSize)
    
    let gradients = out.gradients(delta: error, wrt: input)
    
    XCTAssertNotNil(gradients.input[safe: 0])
    
    let wrtInput = gradients.input[safe: 0]!
    
    XCTAssertEqual(TensorSize(array: wrtInput.shape), inputSize)
    
    let expectedGradient = Tensor.fillWith(value: 0.1 / (inputSize.rows.asTensorScalar * inputSize.columns.asTensorScalar),
                                           size: inputSize)
    
    let flatGradient = expectedGradient.value.flatten()
    
    for g in flatGradient {
      XCTAssertEqual(g,  0.1 / (inputSize.rows.asTensorScalar * inputSize.columns.asTensorScalar), accuracy: 0.0001)
    }
  }
  
  func testResNetDecodeEncode() {
    let inputSize: TensorSize = .init(rows: 16, columns: 16, depth: 1)

    let resNet = ResNet(inputSize: inputSize, filterCount: 8, stride: 1)
    
    let expectedWeights = resNet.weights
    
    do {
      let jsonOut = try JSONEncoder().encode(resNet)
      let jsonIn = try JSONDecoder().decode(ResNet.self, from: jsonOut)
      
      let outWeights = jsonIn.weights
      
      XCTAssertTrue(expectedWeights.isValueEqual(to: outWeights))
    } catch {
      XCTFail(error.localizedDescription)
    }
  }
  
  func testResNet() {
    let inputSize: TensorSize = .init(rows: 64, columns: 64, depth: 3)

    let resNet = ResNet(inputSize: inputSize, filterCount: 16, stride: 2)
    
    let outputSize = resNet.outputSize

    let input = Tensor.fillRandom(size: inputSize)
    
    let out = resNet.forward(tensor: input, context: .init())
        
    XCTAssertEqual(out.shape, outputSize.asArray)
    
    let error = Tensor.fillRandom(size: outputSize)
    
    let gradients = out.gradients(delta: error, wrt: input)
    
    XCTAssertNotNil(gradients.input.first)

    XCTAssertEqual(gradients.input.first?.shape, inputSize.asArray)
    
    // validate it doesn't crash basically...
    resNet.apply(gradients: (Tensor(), Tensor()), learningRate: 0.01)
  }
  
  func test_invalid_input_size() {
    let sequential = Sequential {
      [
        ReLu(inputSize: .init(array: [1,1,0])),
        ReLu()
      ]
    }
    
    sequential.compile()
    
    XCTAssertFalse(sequential.isCompiled)
  }
  
  func test_backpropagation_wrt() {
    
    // first branch
    let dense_0 = Dense(20,
                      inputs: 10,
                      initializer: .heNormal,
                      biasEnabled: true)
    
    let dense = Dense(20,
                      inputs: 20,
                      initializer: .heNormal,
                      biasEnabled: true)
    
    let relu = ReLu(inputSize: dense.outputSize)
    
    
    // second branch
    let dense2 = Dense(20,
                       inputs: 10,
                       initializer: .heNormal,
                       biasEnabled: true)
    
    
    let relu2 = ReLu(inputSize: dense2.outputSize)
    
    // output branch
    let dense3 = Dense(30,
                       inputs: relu2.outputSize.columns,
                       initializer: .heNormal,
                       biasEnabled: true)
    
    let relu3 = ReLu(inputSize: dense3.outputSize)
    
    /*
       input_1
         |
       Dense0  input_2
         |       |
       Dense1  Dense2
         |       |
       Relu1   Relu2
          \     /
           \   /
           Dense3 (dual input graph built here)
          /     \
        Relu3   out_2 (current not used)
         |
        out_1 (gradients calculated here)
     
    1. when getting gradients wrt to input_2 at `out` we shouldn't get anything because the
    output of that branch wasn't used at `out`
     
    2. Figure out how when passing twice we set the same graph twice
     */
    
    // feed forward
    let inputAtDense0 = Tensor.fillWith(value: 1, size: dense_0.inputSize)
    inputAtDense0.label = "input_1"
    
    let dense0Out = dense_0(inputAtDense0)
    let dense1Out = dense(dense0Out)
    let reluOut1 = relu(dense1Out)
    
    let inputAtDense2 = Tensor.fillWith(value: 0.8, size: dense2.inputSize)
    inputAtDense2.label = "input_2"

    let reluOut2 = relu2(dense2(inputAtDense2))
    
    let dense3Out1 = dense3(reluOut1)
    
    dense3Out1.setGraph(reluOut1)
    dense3Out1.setGraph(reluOut2)

    let out1 = relu3(dense3Out1) // branch_1 out

    let out2 = dense3(reluOut2) // branch_2 out
    
    // branch 1 backwards
    let branch1Error = Tensor.fillWith(value: 0.5, size: relu3.outputSize)
    let branch1Backwards = out1.gradients(delta: branch1Error, wrt: inputAtDense0)
    
    XCTAssertEqual(branch1Backwards.input.count, 5)
    XCTAssertEqual(branch1Backwards.input[0].shape, dense_0.inputSize.asArray)
    
    // branch 2 backwards
    let branch2Error = Tensor.fillWith(value: 0.5, size: dense3.outputSize)
    let branch2Backwards = out2.gradients(delta: branch2Error, wrt: inputAtDense2)
    
    XCTAssertEqual(branch2Backwards.input.count, 3)
    XCTAssertEqual(branch2Backwards.input[0].shape, dense2.inputSize.asArray)
  
  }
  
  func test_gelu() {
    let gelu = GeLu()
    gelu.inputSize = TensorSize(rows: 3, columns: 3, depth: 3)
    
    let input = Tensor.fillWith(value: 1, size: gelu.inputSize)
    
    let output = gelu.forward(tensor: input)
    
    let expected = Tensor.fillWith(value: 0.8413447, size: gelu.inputSize)
    XCTAssertEqual(expected, output)
    
    let error = Tensor.fillWith(value: 0.2, size: gelu.inputSize)

    let errorOut = output.gradients(delta: error)
    
    let expectedDer = Tensor.fillWith(value: 0.22427435, size: gelu.inputSize)
    XCTAssertEqual(expectedDer, errorOut.input[0])
  }
  
  func test_encode_normal_initializer_type_keepsValue() {
    let expectedStd: Tensor.Scalar = 0.1
    let rawInitializer: InitializerType = .normal(std: expectedStd)
    let initializer = rawInitializer.build()
    
    let encoder = JSONEncoder()
    let data = try? encoder.encode(initializer)
    
    XCTAssertNotNil(data)
    
    let decoder = JSONDecoder()
    let newInitializer = try? decoder.decode(Initializer.self, from: data!)
    
    XCTAssertNotNil(newInitializer)
    
    switch newInitializer!.type {
    case .normal(let std):
      XCTAssertEqual(std, expectedStd, accuracy: 0.00001)
    default:
      XCTFail("Incorrect initializer decoded")
    }
  }
  
  func test_encode_initializers() {
    let rawInitializer: InitializerType = .heNormal
    let initializer = rawInitializer.build()
    
    let encoder = JSONEncoder()
    let data = try? encoder.encode(initializer)
    
    XCTAssertNotNil(data)
    
    let decoder = JSONDecoder()
    let newInitializer = try? decoder.decode(Initializer.self, from: data!)
    
    XCTAssertNotNil(newInitializer)
    XCTAssertEqual(initializer.type, newInitializer!.type)
  }
  
  // MARK: Sequential
  func test_sequential_importExport_Compressed() {
    
    let size = TensorSize(array: [28,28,1])

    let initializer: InitializerType = .heNormal
    
    let firstLayerFilterCount = 8
    let firstDenseLayerDepthCount = firstLayerFilterCount
    let denseLayerOutputSize = (size.columns / 4, size.rows / 4, firstLayerFilterCount)
    let denseLayerOutputCount = denseLayerOutputSize.0 * denseLayerOutputSize.1 * firstDenseLayerDepthCount
    
    let n = Sequential {
      [
        Dense(denseLayerOutputCount,
              inputs: 100,
              initializer: initializer,
              biasEnabled: false),
        LeakyReLu(limit: 0.2),
        Reshape(to: [size.columns / 4, size.rows / 4, firstDenseLayerDepthCount].tensorSize),
        TransConv2d(filterCount: firstLayerFilterCount * 2, //14x14
                    strides: (2,2),
                    padding: .same,
                    filterSize: (3,3),
                    initializer: initializer,
                    biasEnabled: false),
        LeakyReLu(limit: 0.2),
        TransConv2d(filterCount: firstLayerFilterCount, //28x28
                    strides: (2,2),
                    padding: .same,
                    filterSize: (3,3),
                    initializer: initializer,
                    biasEnabled: false),
        LeakyReLu(limit: 0.2),
        Conv2d(filterCount: size.depth,
               strides: (1,1),
               padding: .same,
               filterSize: (7,7),
               initializer: initializer,
               biasEnabled: false),
        Tanh()
      ]
    }
    
    n.compile()

    guard let gUrl = ExportHelper.getModel(filename: "generator", compress: true, model: n) else {
      XCTFail("invalid URL")
      return
    }
    
    let newN = Sequential.import(gUrl)
    newN.compile()
    
    XCTAssertEqual(newN.debugDescription, n.debugDescription)
  }
  
  func test_sequential_importExport_not_Compressed() {
    
    let size = TensorSize(array: [28,28,1])
    
    let initializer: InitializerType = .heNormal
    
    let firstLayerFilterCount = 8
    let firstDenseLayerDepthCount = firstLayerFilterCount
    let denseLayerOutputSize = (size.columns / 4, size.rows / 4, firstLayerFilterCount)
    let denseLayerOutputCount = denseLayerOutputSize.0 * denseLayerOutputSize.1 * firstDenseLayerDepthCount
    
    let n = Sequential {
      [
        Dense(denseLayerOutputCount,
              inputs: 100,
              initializer: initializer,
              biasEnabled: false),
        LeakyReLu(limit: 0.2),
        Reshape(to: [size.columns / 4, size.rows / 4, firstDenseLayerDepthCount].tensorSize),
        TransConv2d(filterCount: firstLayerFilterCount * 2, //14x14
                    strides: (2,2),
                    padding: .same,
                    filterSize: (3,3),
                    initializer: initializer,
                    biasEnabled: false),
        LeakyReLu(limit: 0.2),
        TransConv2d(filterCount: firstLayerFilterCount, //28x28
                    strides: (2,2),
                    padding: .same,
                    filterSize: (3,3),
                    initializer: initializer,
                    biasEnabled: false),
        LeakyReLu(limit: 0.2),
        Conv2d(filterCount: size.depth,
               strides: (1,1),
               padding: .same,
               filterSize: (7,7),
               initializer: initializer,
               biasEnabled: false),
        Tanh()
      ]
    }
    
    n.compile()

    guard let gUrl = ExportHelper.getModel(filename: "generator", compress: false, model: n) else {
      XCTFail("invalid URL")
      return
    }
    
    let newN = Sequential.import(gUrl)
    newN.compile()
    
    XCTAssertEqual(newN.debugDescription, n.debugDescription)
    
  }
  
  func test_Sequential_importWeights() {
    let network = Sequential {
      [
        Dense(5,
              inputs: 5,
              initializer: .heNormal,
              biasEnabled: true),
        ReLu(),
        Dense(5, initializer: .heNormal,
              biasEnabled: true),
        ReLu()
      ]
    }
    
    network.compile()
    
    do {
      let newWeights = try network.exportWeights().map { $0.map { $0.zerosLike() }}
      try network.importWeights(newWeights)
      try network.exportWeights().forEach { $0.forEach { XCTAssertTrue($0.isValueEqual(to: $0.zerosLike() ))}}
    } catch {
      XCTFail(error.localizedDescription)
    }
  }
  
  func test_Sequential_exportWeights_didNotCompile() {
    let network = Sequential {
      [
        Dense(20,
              inputs: 8,
              initializer: .heNormal,
              biasEnabled: true),
        ReLu(),
        Dense(10, initializer: .heNormal,
              biasEnabled: true),
        ReLu()
      ]
    }
        
    do {
      let _ = try network.exportWeights().map { $0.map { $0.zerosLike() }}
    } catch {
      XCTAssertTrue(true)
    }
  }
  
  // MARK: SeLu
  func test_seLu() {
    let input: [[[Tensor.Scalar]]] = [[[0.0, 1.0, -1.0, 0.0],
                               [0.0, 1.0, -1.0, 0.0]],
                              [[0.0, 1.0, -1.0, 0.0],
                               [0.0, 1.0, -1.0, 0.0]]]
    
    let inputSize = input.shape

    let layer = SeLu(inputSize: TensorSize(array: inputSize))
    let out = layer.forward(tensor: Tensor(input))
    
    XCTAssertEqual(inputSize, out.shape)
    
    let expected: [[[Tensor.Scalar]]] = [[[0.0, 1.0507, -1.1113541, 0.0],
                                 [0.0, 1.0507, -1.1113541, 0.0]],
                                [[0.0, 1.0507, -1.1113541, 0.0],
                                 [0.0, 1.0507, -1.1113541, 0.0]]]
    
    XCTAssertEqual(expected, out.value)
    
    let delta = Tensor([[[-1.0, 1.0, -1.0, 0.0],
                         [-1.0, 1.0, -1.0, 0.0]],
                        [[-1.0, 1.0, -1.0, 0.0],
                         [-1.0, 1.0, -1.0, 0.0]]])
    
    let gradients = out.gradients(delta: delta)
        
    let expectedGradients: [[[Tensor.Scalar]]] = [[[-1.7581363, 1.0507, -0.6467822, 0.0],
                                           [-1.7581363, 1.0507, -0.6467822, 0.0]],
                                          [[-1.7581363, 1.0507, -0.6467822, 0.0],
                                           [-1.7581363, 1.0507, -0.6467822, 0.0]]]
    
    XCTAssertEqual(gradients.input.first!.value, expectedGradients)
  }
  
  // MARK: AvgPool
  func test_avgPool_7x7_kernel_size() {
    let input: [[[Tensor.Scalar]]] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.3, 1.4, 1.5].as3D()
    let inputSize = input.shape

    let layer = AvgPool(inputSize: TensorSize(array: inputSize), kernelSize: (7,7))
    let out = layer.forward(tensor: Tensor(input))
    
    XCTAssertEqual([2,2,14], out.shape)
    
    let expected: [[[Tensor.Scalar]]] = [[[Tensor.Scalar]]].init(repeating: [[0.15, 0.85],
                                                                             [0.15, 0.85]], count: 14)
    
    XCTAssertEqual(expected, out.value)
    
    let delta: [[[Tensor.Scalar]]] = [[[Tensor.Scalar]]].init(repeating: [[0.1, 0.3],
                                                                          [0.1, 0.3]], count: 14)
    
    let gradients = out.gradients(delta: Tensor(delta))
    
    XCTAssertEqual(inputSize, gradients.input.first!.shape)
  }
  
  
  func test_avgPool() {
    let input: [[[Tensor.Scalar]]] = [[[0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4]],
                              [[0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4]],
                              [[0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4]]]
    
    let inputSize = input.shape

    let layer = AvgPool(inputSize: TensorSize(array: inputSize))
    let out = layer.forward(tensor: Tensor(input))
    
    XCTAssertEqual([2,2,3], out.shape)
    
    let expected: [[[Tensor.Scalar]]] = [[[0.15, 0.35],
                                  [0.15, 0.35]],
                                 [[0.15, 0.35],
                                  [0.15, 0.35]],
                                 [[0.15, 0.35],
                                  [0.15, 0.35]]]
    
    XCTAssertEqual(expected, out.value)
    
    let delta: [[[Tensor.Scalar]]] = [[[0.1, 0.3],
                               [0.2, 0.5]],
                              [[0.1, 0.3],
                               [0.2, 0.5]],
                              [[0.1, 0.3],
                               [0.2, 0.5]]]
    
    let gradients = out.gradients(delta: Tensor(delta))
    
    XCTAssertEqual(inputSize, gradients.input.first!.shape)
    
    let expectedGradients: [[[Tensor.Scalar]]] = [[[0.025, 0.025, 0.075, 0.075],
                                           [0.025, 0.025, 0.075, 0.075],
                                           [0.05, 0.05, 0.125, 0.125],
                                           [0.05, 0.05, 0.125, 0.125]],
                                          [[0.025, 0.025, 0.075, 0.075],
                                           [0.025, 0.025, 0.075, 0.075],
                                           [0.05, 0.05, 0.125, 0.125],
                                           [0.05, 0.05, 0.125, 0.125]],
                                          [[0.025, 0.025, 0.075, 0.075],
                                           [0.025, 0.025, 0.075, 0.075],
                                           [0.05, 0.05, 0.125, 0.125],
                                           [0.05, 0.05, 0.125, 0.125]]]

    XCTAssertEqual(expectedGradients, gradients.input.first!.value)
  }
  
  // MARK: Dense
  func test_Dense_Parameters() {
    let dense = Dense(20,
                      inputs: 8,
                      initializer: .heNormal,
                      biasEnabled: true)
    
    XCTAssertEqual(dense.biases.shape, [1, 1, 1])
  }
  
  func test_Dense_importWeights_valid() {
    let dense = Dense(20,
                      inputs: 8,
                      initializer: .heNormal,
                      biasEnabled: true)
    
    do {
      let newWeights = try dense.exportWeights()[safe: 0, Tensor()].zerosLike()
      try dense.importWeights([newWeights])
      XCTAssert(try dense.exportWeights().first!.isValueEqual(to: newWeights))
    } catch {
      XCTFail(error.localizedDescription)
    }
  }
  
  func test_Dense_importWeights_invalid_Shape() {
    let dense = Dense(256,
                      inputs: 100,
                      initializer: .heNormal,
                      biasEnabled: true)
    
    do {
      try dense.importWeights([Tensor([10, 10, 10])])
    } catch {
      if let _ = error as? LayerErrors {
        XCTAssertTrue(true)
      } else {
        XCTFail()
      }
    }
  }
  
  // MARK: Convolution
  func test_Conv2d_filters() {
    let conv = Conv2d(filterCount: 32,
                      inputSize: .init(array: [28,28,8]),
                      padding: .same,
                      filterSize: (3,3),
                      initializer: .heNormal,
                      biasEnabled: true)
    
    XCTAssertFalse(conv.filters.isEmpty)
    XCTAssertEqual(conv.filters.shape, [32])
    conv.filters.forEach { f in
      XCTAssertEqual(f.shape, [3,3,8])
    }
    
    XCTAssertEqual(conv.outputSize, TensorSize(array: [28, 28, 32]))
  }
  
  func test_Conv2d_importWeights_valid() {
    let layer = Conv2d(filterCount: 5,
                       inputSize: .init(array: [28,28,1]),
                       filterSize: (3,3),
                       initializer: .heNormal)
    
    do {
      let newWeights = try layer.exportWeights().map { $0.zerosLike() }
      try layer.importWeights(newWeights)
      let exported = try layer.exportWeights()
      
      for i in 0..<exported.count {
        let export = exported[i]
        let new = newWeights[i]
        XCTAssert(new.isValueEqual(to: export))
      }
    } catch {
      XCTFail(error.localizedDescription)
    }
  }
  
  func test_Conv2d_importWeights_invalid_Shape() {
    let layer = Conv2d(filterCount: 5,
                       inputSize: .init(array: [28,28,1]),
                       filterSize: (3,3),
                       initializer: .heNormal)
    
    do {
      try layer.importWeights([Tensor([10, 10, 10])])
    } catch {
      if let _ = error as? LayerErrors {
        XCTAssertTrue(true)
      } else {
        XCTFail()
      }
    }
  }
  
  func test_TransConv2d_filters() {
    let conv = TransConv2d(filterCount: 32,
                           inputSize: .init(array: [28,28,8]),
                           strides: (2,2),
                           padding: .same,
                           filterSize: (3,3),
                           initializer: .heNormal,
                           biasEnabled: true)
    
    XCTAssertFalse(conv.filters.isEmpty)
    XCTAssertEqual(conv.filters.shape, [32])
    conv.filters.forEach { f in
      XCTAssertEqual(f.shape, [3,3,8])
    }
    
    XCTAssertEqual(conv.outputSize, TensorSize(array: [56, 56, 32]))
  }

  
  // MARK: LSTM
  
  func test_LSTM_Weights() {
    let inputUnits = 100
    let hiddenUnits = 256
    let vocabSize = 27
    
    let lstm = LSTM(inputUnits: inputUnits,
                    batchLength: 1,
                    hiddenUnits: hiddenUnits,
                    vocabSize: vocabSize)
    
    XCTAssertEqual(lstm.hiddenOutputWeights.shape, [hiddenUnits, vocabSize, 1])
    
    XCTAssertEqual(lstm.forgetGateWeights.shape, [hiddenUnits, hiddenUnits + inputUnits, 1])
    XCTAssertEqual(lstm.inputGateWeights.shape, [hiddenUnits, hiddenUnits + inputUnits, 1])
    XCTAssertEqual(lstm.outputGateWeights.shape, [hiddenUnits, hiddenUnits + inputUnits, 1])
    XCTAssertEqual(lstm.gateGateWeights.shape, [hiddenUnits, hiddenUnits + inputUnits, 1])
  }
  
  func test_LSTM_Forward() {
    let names = ["anna",
                 "emma",
                 "elizabeth",
                 "minnie",
                 "margaret",
                 "ida",
                 "alice",
                 "bertha",
                 "sarah"]
    
    let vectorizer = Vectorizer<String>()

    let batchLength = 10
    
    names.forEach { name in
      vectorizer.vectorize(name.fill(with: ".", max: batchLength).characters)
    }
    
    let testName = "anna".fill(with: ".", max: batchLength)
    let oneHot = vectorizer.oneHot(testName.characters)
    
    let inputUnits = 10
    let hiddenUnits = 256
    let vocabSize = vectorizer.vector.count // the size of the total map of vocab letters available. Likely comes from Vectorize
    let inputTensor = oneHot

    let embedding = Embedding(inputUnits: inputUnits,
                              vocabSize: vocabSize,
                              batchLength: batchLength)
    
    let embeddingCalc = embedding.forward(tensor: inputTensor)

    let lstm = LSTM(inputUnits: inputUnits,
                    batchLength: batchLength,
                    initializer: .heNormal,
                    hiddenUnits: hiddenUnits,
                    vocabSize: vocabSize)

        
    let out = lstm.forward(tensor: embeddingCalc, context: .init())
    
    XCTAssertEqual(out.shape, [vocabSize, 1, batchLength])
  }

  func test_Embedding_Forward() {
    let names = ["anna",
                 "emma",
                 "elizabeth",
                 "minnie",
                 "margaret",
                 "ida",
                 "alice",
                 "bertha",
                 "sarah"]
    
    let vectorizer = Vectorizer<String>()

    let batchLength = 10
    
    names.forEach { name in
      vectorizer.vectorize(name.fill(with: ".", max: batchLength).characters)
    }
    
    let testName = "anna".fill(with: ".", max: batchLength)
    let oneHot = vectorizer.oneHot(testName.characters)
    
    let inputUnits = 100
    let vocabSize = vectorizer.vector.count
    
    let embedding = Embedding(inputUnits: inputUnits,
                              vocabSize: vocabSize,
                              batchLength: batchLength)
    
    let inputTensor = oneHot
    
    let out = embedding.forward(tensor: inputTensor)
    
    XCTAssertEqual(out.shape, [inputUnits, 1, batchLength])
  }

}
