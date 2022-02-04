//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/21/21.
//

import Foundation


public struct TestConstants {
  public static let inputs = 4 //UIColor values rgba
  public static let hidden = 5
  public static let outputs = ColorType.allCases.count
  public static let numOfHiddenLayers = 1
  public static let lossThreshold: Float = 0.1
  public static let testingLossThreshold: Float = 0.25 //if below 0.2 considered trained
  public static let samplePretrainedModel: String = "sample_color_model"
  public static let samplePretrainedModelNormalized: String = "sample_color_model_normalized"
}
