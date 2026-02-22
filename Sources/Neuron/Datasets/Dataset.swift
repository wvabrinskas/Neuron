//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/24/22.
//

import Foundation
import Combine

/// A model representing a single supervised learning sample consisting of input data and its corresponding label.
public struct DatasetModel: Equatable {
  /// The input feature tensor for this dataset sample.
  public var data: Tensor
  /// The ground-truth label tensor for this dataset sample.
  public var label: Tensor
  
  /// Creates one supervised dataset sample.
  ///
  /// - Parameters:
  ///   - data: Input feature tensor.
  ///   - label: Ground-truth label tensor.
  public init(data: Tensor, label: Tensor) {
    self.data = data
    self.label = label
  }
}

