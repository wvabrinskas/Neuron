//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/18/21.
//

import Foundation

public enum Inializers {
  case xavierNormal, xavierUniform
  
  public func calculate(m: Int, h: Int) -> Float {
    switch self {
    
    case .xavierNormal:
      let range = m > h ? Float(h)...Float(m) : Float(m)...Float(h)
      let random = Float.random(in: range)
      return random * Float(sqrt(2 / (Double(m) + Double(h))))
      
    case .xavierUniform:
      let random = Float.random(in: -1...1)
      return random * Float(sqrt(6.0 / (Double(m) + Double(h))))
    }
    
  
  }
}
