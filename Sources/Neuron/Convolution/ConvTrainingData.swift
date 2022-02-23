//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/23/22.
//

import Foundation


public struct ConvTrainingData {
  public var data: [[[Float]]]
  public var label: [Float]
  
  public init(data: [[[Float]]], label: [Float]) {
    self.data = data
    self.label = label
  }
}
