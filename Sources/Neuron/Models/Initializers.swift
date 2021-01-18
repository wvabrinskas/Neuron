//
//  Created by William Vabrinskas on 1/18/21.
//

import Foundation
import GameplayKit

/// Weight initializer methods
public enum Initializers {
  ///Generates weights based on a normal gaussian distribution. Mean = 0 sd = 1
  case xavierNormal
  ///Generates weights based on a uniform distribution
  case xavierUniform
  
  /// Calculates a weight given input and output node counts
  /// - Parameters:
  ///   - m: Node count in or out
  ///   - h: Node count in or out
  /// - Returns: Weight calculated based on type of initializer
  public func calculate(m: Int, h: Int) -> Float {
    switch self {
    
    case .xavierUniform:
      let range = m > h ? Float(h)...Float(m) : Float(m)...Float(h)
      let random = Float.random(in: range)
      return random * Float(sqrt(2 / (Double(m) + Double(h))))
      
    case .xavierNormal:
      return Brain.dist.nextFloat() * Float(sqrt(6.0 / (Double(m) + Double(h))))
    }
    
  }
}
