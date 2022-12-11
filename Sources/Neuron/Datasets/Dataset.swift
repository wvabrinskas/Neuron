//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/24/22.
//

import Foundation
import Combine

public struct DatasetModel: Equatable {
  public var data: Tensor
  public var label: Tensor
  
  public init(data: Tensor, label: Tensor) {
    self.data = data
    self.label = label
  }
}

