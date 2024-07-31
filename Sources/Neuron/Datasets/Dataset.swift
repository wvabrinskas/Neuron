//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/24/22.
//

import Foundation
import Combine

public struct DatasetModel<N: TensorNumeric>: Equatable {
  public var data: Tensor<N>
  public var label: Tensor<N>
  
  public init(data: Tensor<N>, label: Tensor<N>) {
    self.data = data
    self.label = label
  }
}

