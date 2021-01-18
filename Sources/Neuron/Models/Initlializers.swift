//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/18/21.
//

import Foundation

public enum Inializers {
  case xavier
  
  public func calculate(m: Int, h: Int) -> Float {
    //w=np.random.randn(layer_size[l],layer_size[l-1])*np.sqrt(2/(layer_size[l-1]+layer_size[l]))
    
    let range = m > h ? Float(h)...Float(m) : Float(m)...Float(h)
    let random = Float.random(in: range)
    return random * Float(sqrt(2 / (Double(m) + Double(h))))
  }
}
