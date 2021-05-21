//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/21/21.
//

import Foundation

public struct GANModel {
  public var inputs: Int
  public var hiddenLayers: Int
  public var hiddenNodesPerLayer: Int
  public var outputs: Int
  public var generatorInputs: Int
  
  public init(inputs: Int,
              hiddenLayers: Int,
              hiddenNodesPerLayer: Int,
              outputs: Int,
              generatorInputs: Int) {
    self.inputs = inputs
    self.hiddenLayers = hiddenLayers
    self.hiddenNodesPerLayer = hiddenNodesPerLayer
    self.outputs = outputs
    self.generatorInputs = generatorInputs
  }
}


public class GAN {
  private var generator: Brain
  private var discriminator: Brain
  
  public var randomNoise: () -> [Float]
  
  //create two networks
  //one for the generator
  //one for the discriminator
  // link them using neuron network
  //backprop can skip input layer of discriminator when trainign generator
  //backprop will of generator is as saame as a regular NN
  
 public init(ganModel: GANModel,
             learningRate: Float,
             epochs: Int,
             lossThreshold: Float = 0.001,
             initializer: Initializers = .xavierNormal,
             descent: GradientDescent = .sgd) {
    //generator
    let brainGen = Brain(learningRate: learningRate,
                         epochs: epochs,
                         lossFunction: .crossEntropy,
                         lossThreshold: lossThreshold,
                         initializer: initializer,
                         descent: descent)
    
    brainGen.add(LobeModel(nodes: ganModel.generatorInputs))
    
    for _ in 0..<ganModel.hiddenLayers {
      brainGen.add(LobeModel(nodes: ganModel.hiddenNodesPerLayer, activation: .reLu))
    }
    
    brainGen.add(LobeModel(nodes: ganModel.outputs, activation: .reLu))
    
    self.generator = brainGen
    self.generator.compile()
    
    //discriminator
    let brainDis = Brain(learningRate: learningRate,
                         epochs: epochs,
                         lossFunction: .crossEntropy,
                         lossThreshold: lossThreshold,
                         initializer: initializer,
                         descent: descent)
    
    //inputs of discrimnator should be the same as outputs of the generator
    brainDis.add(LobeModel(nodes: ganModel.outputs)) //input
    for _ in 0..<ganModel.hiddenLayers {
      brainGen.add(LobeModel(nodes: ganModel.hiddenNodesPerLayer, activation: .leakyRelu))
    }
    brainGen.add(LobeModel(nodes: ganModel.outputs, activation: .none)) //output

    self.discriminator = brainDis
    self.discriminator.compile()
    
    self.randomNoise = {
      var noise: [Float] = []
      for _ in 0..<ganModel.generatorInputs {
        noise.append(Float.random(in: 0...1))
      }
      return noise
    }
  }

  
  private func buildLink() {
    //link generator and discriminator together
    
  }
  
  public func getGeneratedSample() -> [Float] {
    //need to feed generator random noise
    //return generator output after training
    //probably will be the size of the output
    //add function that will generator random noise
    return generator.feed(input: randomNoise())
  }
  
}
