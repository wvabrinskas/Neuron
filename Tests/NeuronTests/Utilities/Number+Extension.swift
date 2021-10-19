//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/21/21.
//

import Foundation

protocol Numeric {
  var asDouble: Double { get }
  init(_: Double)
}

extension Int: Numeric {var asDouble: Double { get {return Double(self)}}}
extension Float: Numeric {var asDouble: Double { get {return Double(self)}}}
extension Double: Numeric {var asDouble: Double { get {return Double(self)}}}
extension CGFloat: Numeric {var asDouble: Double { get {return Double(self)}}}

extension Array where Element: Numeric {
  
  var sd : Element { get {
    let sss = self.reduce((0.0, 0.0)){ return ($0.0 + $1.asDouble, $0.1 + ($1.asDouble * $1.asDouble))}
    let n = Double(self.count)
    return Element(sqrt(sss.1/n - (sss.0/n * sss.0/n)))
  }}
  
  
  var mean : Element { get {
    let sum = self.reduce(0.asDouble, { x, y in
      x.asDouble + y.asDouble
    })
    
    return Element(sum / Double(self.count))
  }}
}
