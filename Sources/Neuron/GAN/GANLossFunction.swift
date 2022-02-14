//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/11/22.
//

import Foundation

public enum GANType {
  case generator, discriminator
}

public enum GANTrainingType: String {
  case real, fake
}

public enum GANLossFunction {
  case minimax
  case wasserstein
  
  public func label(type: GANTrainingType) -> Float {
    switch self {
    case .minimax:
      switch type {
      case .real:
        return 1.0
      case .fake:
        return 0
      }
      
    case .wasserstein:
      switch type {
      case .real:
        return 1
      case .fake:
        return -1
      }
    }
  }
  
  public func loss(_ type: GANTrainingType, value: Float) -> Float {
    switch self {
    case .minimax:
      switch type {
      case .fake:
        return log(1 - value)
      case .real:
        return log(value)
      }
    case .wasserstein:
      return self.label(type: type) * value
    }
  }
}
