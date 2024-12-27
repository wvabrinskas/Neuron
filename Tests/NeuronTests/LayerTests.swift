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
  
  func test_backpropagation() {
    
    // first branch
    let dense = Dense(20,
                      inputs: 10,
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
       Dense1  Dense2
         |       |
       Relu1   Relu2
          \     /
           \   /
           Dense3 (dual input graph built here)
          /     \
        Relu3   (current not used)
         |
        out (gradients calculated here)
     */
    
    // feed forward
    let inputAtDense1 = Tensor.fillWith(value: 1, size: dense.inputSize)
    
    let reluOut1 = relu(dense(inputAtDense1))
    
    let inputAtDense2 = Tensor.fillWith(value: 0.8, size: dense2.inputSize)
    
    let reluOut2 = relu2(dense2(inputAtDense2))
    
    let dense3Out1 = dense3(reluOut1)
    let dense3Out2 = dense3(reluOut2)
    
    dense3Out1.setGraph(reluOut2)

    let out = relu3(dense3Out1)
        
    print(out)
    
    // full backward
    let error = Tensor.fillWith(value: 0.5, size: relu3.outputSize)
    
    let backwards = out.gradients(delta: error)
    
    print(backwards)
    
    // single branch backward
    let dense3BranchError = Tensor.fillWith(value: 0.5, size: dense3.outputSize)
    let dense3BackwardsBranch = dense3Out2.gradients(delta: dense3BranchError)
    print(dense3BackwardsBranch)

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
