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
  
  // MARK: LSTM
  
  func test_LSTM_Weights() {
    let inputUnits = 100
    let hiddenUnits = 256
    let vocabSize = 27
    
    let lstm = LSTM(inputSize: TensorSize(rows: 1, columns: inputUnits, depth: 1),
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

    names.forEach { name in
      vectorizer.vectorize(name.fill(with: ".", max: 10).characters)
    }
    
    let testName = "anna".fill(with: ".", max: 10)
    let oneHot = vectorizer.oneHot(testName.characters)
    
    let inputUnits = 10
    let hiddenUnits = 256
    let vocabSize = vectorizer.vector.count // the size of the total map of vocab letters available. Likely comes from Vectorize

    let lstm = LSTM(inputSize: TensorSize(rows: oneHot.count,
                                          columns: inputUnits,
                                          depth: 1),
                    initializer: .heNormal,
                    hiddenUnits: hiddenUnits,
                    vocabSize: vocabSize)

    
    let inputTensor = Tensor(oneHot)
    
    let out = lstm.forward(tensor: inputTensor)
    
    XCTAssertEqual(out.shape, [vocabSize, 1, oneHot.count])
  }

}
