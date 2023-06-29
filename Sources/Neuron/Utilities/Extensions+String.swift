//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/20/23.
//

import Foundation

public extension String {
  var characters: [String] {
    map { String($0) }
  }

  func fill(with char: Character = " ", max: Int, leftAlign: Bool = true) -> String {
    guard max - self.count > 0 else {
      return self
    }
    
    var mapped = self
    for _ in 0..<(max - self.count) {
      let index = leftAlign ? self.endIndex : self.startIndex
      mapped.insert(char, at: index)
    }
    return mapped
  }
}
