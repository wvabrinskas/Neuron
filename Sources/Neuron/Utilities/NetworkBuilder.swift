//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/21/21.
//

import Foundation

public protocol NetworkBuilder {
  init(model: ExportModel?,
        learningRate: Float,
        epochs: Int,
        lossFunction: LossFunction,
        lossThreshold: Float,
        initializer: Initializers,
        descent: GradientDescent)
}
