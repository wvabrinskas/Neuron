//
//  File.swift
//  
//
//  Created by William Vabrinskas on 3/17/22.
//

import Foundation
import SwiftUI

public struct BrainViewModel {
  var lobes: [LobeViewModel]
  var spacing: CGFloat
  
  public init(lobes: [LobeViewModel], spacing: CGFloat = 40) {
    self.lobes = lobes
    self.spacing = spacing
  }
}
