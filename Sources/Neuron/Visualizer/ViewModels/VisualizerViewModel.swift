//
//  File.swift
//  
//
//  Created by William Vabrinskas on 3/17/22.
//

import Foundation
import SwiftUI

public struct VisualizerViewModel {
  public var brain: BrainViewModel
  
  public init(brain: BrainViewModel = .init()) {
    self.brain = brain
  }
}
