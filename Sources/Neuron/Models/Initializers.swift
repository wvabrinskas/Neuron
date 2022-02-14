//
//  Created by William Vabrinskas on 1/18/21.
//

import Foundation
import GameplayKit

/// Weight initializer methods
public enum Initializers: String, Codable {
  ///Generates weights based on a normal gaussian distribution. Mean = 0 sd = 1
  case xavierNormal
  ///Generates weights based on a uniform distribution
  case xavierUniform
  
  case heNormal
  case heUniform
  
  /// Calculates a weight given input and output node counts
  /// - Parameters:
  ///   - input: Node count in
  ///   - out: Node count out
  /// - Returns: Weight calculated based on type of initializer
  public func calculate(input: Int, out: Int = 0) -> Float {
    switch self {
      
    case .xavierUniform:
      let min = -Float(sqrt(6) / sqrt((Double(input) + Double(out))))
      let max = Float(sqrt(6) / sqrt((Double(input) + Double(out))))
      
      return Float.random(in: min...max)
      
    case .xavierNormal:
      return Brain.dist.nextFloat() * Float(sqrt(2 / (Double(input) + Double(out))))
      
    case .heUniform:
      let min = -Float(sqrt(6) / sqrt((Double(input))))
      let max = Float(sqrt(6) / sqrt((Double(input))))
      
      return Float.random(in: min...max)
      
    case .heNormal:
      return Brain.dist.nextFloat() * Float(sqrt(2 / (Double(input))))
    }
    
  }
}
