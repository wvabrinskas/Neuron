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
  
  // MARK: Sequential
  func test_sequential_importExport() {
    
    let size = TensorSize(array: [64,64,3])
    
    let initializer: InitializerType = .heNormal
    
    let firstLayerFilterCount = 32
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

    guard let gUrl = ExportHelper.getModel(filename: "generator", model: n) else {
      XCTAssert(true)
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
    
    XCTAssertEqual(lstm.hiddenOutputWeights.shape, [vocabSize, hiddenUnits, 1])
    
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

        
    let out = lstm.forward(tensor: embeddingCalc)
    
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
