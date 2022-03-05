//
//  Created by William Vabrinskas on 1/18/21.
//

import Foundation
import GameplayKit

/// Weight initializer methods
public enum InitializerType: String, Codable {
  ///Generates weights based on a normal gaussian distribution. Mean = 0 sd = 1
  case xavierNormal
  ///Generates weights based on a uniform distribution
  case xavierUniform
  
  case heNormal
  case heUniform
  
  public func build() -> Initializer {
    Initializer(type: self)
  }
}

public struct Initializer {
  public let type: InitializerType
  private let dist: NormalDistribution = NormalDistribution(mean: 0, deviation: 1)
  
  public init(type: InitializerType) {
    self.type = type
  }
  
  public func calculate(input: Int, out: Int = 0) -> Float {
    switch type {
      
    case .xavierUniform:
      let min = -Float(sqrt(6) / sqrt((Double(input) + Double(out))))
      let max = Float(sqrt(6) / sqrt((Double(input) + Double(out))))
      
      return Float.random(in: min...max)
      
    case .xavierNormal:
      return dist.nextFloat() * Float(sqrt(2 / (Double(input) + Double(out))))
      
    case .heUniform:
      let min = -Float(sqrt(6) / sqrt((Double(input))))
      let max = Float(sqrt(6) / sqrt((Double(input))))
      
      return Float.random(in: min...max)
      
    case .heNormal:
      return dist.nextFloat() * Float(sqrt(2 / (Double(input))))
    }
    
  }
}
