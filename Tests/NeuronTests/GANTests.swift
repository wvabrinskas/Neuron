//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/13/22.
//

import Foundation
import XCTest
import GameKit
@testable import Neuron

final class GANTests: XCTestCase {
  private let dist = NormalDistribution(mean: 0, deviation: 0.2)
  private let gaussianInt = GKGaussianDistribution(lowestValue: 0, highestValue: 10)
  
  private let mapRange: ClosedRange<Float> = -1...1
  private let gaussianRange: ClosedRange<Float> = 0...10
  private let generatorInputs = 1
  private let wordLength: Int = 1
  
  private lazy var generator: Brain = {
    let bias: Float = 0
    
    let brain = Brain(learningRate: 0.00001,
                      epochs: 1)
    
    brain.add(.init(nodes: generatorInputs))
    brain.add(.init(nodes: 10, activation: .leakyRelu, bias: bias))
    // brain.add(.init(nodes: 5, activation: .leakyRelu, bias: bias))
    brain.add(.init(nodes: wordLength, activation: .tanh, bias: bias))
    
    brain.logLevel = .none
    brain.add(optimizer: .adam())
    brain.compile() //build network
    
    return brain
  }()
  
  private lazy var discriminator: Brain = {
    let bias: Float = 0
    let brain = Brain(learningRate: 0.00001,
                      epochs: 1)
    
    brain.add(.init(nodes: wordLength))
    brain.add(.init(nodes: 10, activation: .leakyRelu, bias: bias))
    // brain.add(.init(nodes: 10, activation: .leakyRelu, bias: bias))
    brain.add(.init(nodes: 1, activation: .sigmoid, bias: bias))
    
    brain.logLevel = .none
    brain.add(optimizer: .adam())
    brain.compile() //build network
    
    return brain
  }()
  
  private lazy var ganBrain: GAN = {
    let gan = GAN(epochs: 1000,
                  criticTrainPerEpoch: 4,
                  generatorTrainPerEpoch: 1,
                  gradientPenaltyCenter: 1,
                  batchSize: 10)
    
    gan.add(generator: self.generator)
    gan.add(discriminator: self.discriminator)
    
    gan.logLevel = .none
    gan.lossFunction = .wasserstein
    
    gan.randomNoise = { [weak self] in
      guard let strongSelf = self else {
        return []
      }
      let out = strongSelf.randomInput(length: strongSelf.generatorInputs,
                                       guassian: false)
      
      return out
    }
    
    gan.validateGenerator = { _ in
      return false
    }
    
    return gan
  }()
  
  
  private func trainingDataFromGaussian() -> [TrainingData] {
    let label = self.ganBrain.lossFunction.label(type: .real)
    
    var gaussianData: [TrainingData] = []
    for _ in 0..<1000 {
      let gaussian = self.randomInput(length: generatorInputs, guassian: true)
      let training = TrainingData(data: gaussian, correct: [label])
      gaussianData.append(training)
    }
    
    return gaussianData
  }
  
  
  func randomInput(length: Int, guassian: Bool = false) -> [Float] {
    var randomInput: [Float] = []
    for _ in 0..<length {
      var rand = Float.random(in: self.mapRange)
      if guassian {
        rand = self.dist.nextFloat()
      }
      
      randomInput.append(rand)
    }
    
    return randomInput
  }
  
  func testTrainGan() {
    let trainingData = self.trainingDataFromGaussian()
    
    self.ganBrain.train(data: trainingData, complete:  { success in
      print(success)
    })
    
  }
  
}
