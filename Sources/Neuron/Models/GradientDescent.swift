//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/27/21.
//

import Foundation

public enum GradientDescent: Equatable {
  case sgd
  case mbgd(size: Int)
  case bgd
}
