//
//  Array+Extensions.swift
//  Nameley
//
//  Created by William Vabrinskas on 12/23/20.
//  Copyright © 2020 William Vabrinskas. All rights reserved.
//

import Foundation


public extension Array where Element: Equatable {
  
  /// Get a copy of self but with randomized data indexes
  /// - Returns: Returns Self but with the data randomized
  func randomize() -> Self {
    var arrayCopy = self
    var randomArray: [Element] = []
    
    for _ in 0..<self.count {
      guard let element = arrayCopy.randomElement() else {
        break
      }
      randomArray.append(element)
      
      if let index = arrayCopy.firstIndex(of: element) {
        arrayCopy.remove(at: index)
      }
    }
    
    return randomArray
  }
}
