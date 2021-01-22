//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/20/21.
//

import Foundation


public struct Layer: Codable {
  var activation: Activation
  var nodes: Int
  var weights: [[Float]]
  var type: LobeModel.LayerType
  var bias: [Float]
  var biasWeights: [Float]
}

public struct ExportModel: Codable {
  public var layers: [Layer]
  public var learningRate: Float
}
