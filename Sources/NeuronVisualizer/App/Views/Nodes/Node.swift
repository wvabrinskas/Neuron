//
//  Node.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/7/24.
//

import SwiftUI

public protocol Node: AnyObject {
  var connections: [Node] { get }
  
  @ViewBuilder
  func build() -> any View
}
