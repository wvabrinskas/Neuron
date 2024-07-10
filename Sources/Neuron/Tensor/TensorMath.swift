//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/27/22.
//

import Foundation
import NumSwift

public extension Tensor {
  typealias MathBlock = (_ feature: [Scalar]) -> Scalar

  /*
       +--------+
      /        /|
     /        Z |
    +---X----+  |
    |        |  |
    |   -1   Y  +
    |        | /
    |        |/
    +--------+
    Along axis 0 the Tensor of shape AxBxC, where A is the columns, B is the rows, and C is the depth, would perform a mathematical function along the Y axis returning a (Ax1xC) Tensor

    Along axis 1 the Tensor of shape AxBxC, where A is the columns, B is the rows, and C is the depth, would perform a mathematical function along the X axis returning a (1xBxC) Tensor

    Along axis 2 the Tensor of shape AxBxC, where A is the columns, B is the rows, and C is the depth, would perform a mathematical function along the Z axis returning a (AxBx1) Tensor

    Along axis -1 the Tensor of shape AxBxC, where A is the columns, B is the rows, and C is the depth, would perform a mathematical function along the Z axis returning a (1x1x1) Tensor Scalar
   */
  func apply(axis: Int, _ block: MathBlock) -> Tensor {
    let shape = shape
    let rows = shape[safe: 1, 0]
    let columns = shape[safe: 0, 0]
    
    var result: [[[Scalar]]] = []
    
    if axis == 0 {
      var featureResults: [[[Scalar]]] = []
      
      for feature in value {
        var resultRow: [Scalar] = []
        for x in 0..<columns {
          var workingRow: [Scalar] = []

          for y in 0..<rows {
            workingRow.append(feature[y][x])
          }
            
          resultRow.append(block(workingRow))
        }
        featureResults.append([resultRow])
      }

      result = featureResults
      
    } else if axis == 1 {
      var featureResults: [[[Scalar]]] = []
      
      for d in 0..<value.count {
        var result: [[Scalar]] = []
        
        for r in 0..<rows {
          result.append([block(value[d][r])])
        }
        
        featureResults.append(result)
      }
                         
      result = featureResults
                        
    } else if axis == 2 {
      var featureResults: [[Scalar]] = []
        
      for r in 0..<rows {
        var results: [Scalar] = []
        for c in 0..<columns {
          
          var featureR: [Scalar] = []
          for d in 0..<value.count {
            let f = value[d][r][c]
            featureR.append(f)
          }
          
          results.append(block(featureR))
        }
        
        featureResults.append(results)
      }
      
      result.append(featureResults)
    }
    
    return Tensor(result)
  }
  
  func clip(_ val: Scalar = 0.01) {
    value = value.map { $0.map { $0.map { Swift.max(-val, Swift.min(val, $0)) }}}
  }
  
  func sum() -> Scalar {
    return value.sum
  }
  
  func testLarge(limit: Scalar) {
    self.value.forEach { t in
      t.forEach { r in
        if r.contains(where: { $0 > limit} ) {
          assertionFailure()
        }
      }
    }
  }
  
  func testNaN() {
    self.value.forEach { t in
      t.forEach { r in
        if r.contains(where: { $0.isNaN} ) {
          assertionFailure()
        }
      }
    }
  }
  
  func matmul(_ with: Tensor) -> Tensor {
    // TODO: maybe do auto grad here?
    let A = value
    let B = with.value
    return Tensor(A.matmul(B))
  }
  
  func sumOfSquares(axis: Int = -1) -> Tensor {
    let block: MathBlock = { feature in
      feature.sumOfSquares
    }
    
    if axis == -1 {
      return Tensor(self.value.sumOfSquares)
    }
    
    return apply(axis: axis, block)
  }
  
  func split(into: Int, axis: Int = 2) -> [Tensor] {    
    if axis == 2 { // along depth
      return self.value.batched(into: into).map { Tensor($0) }
    }
    
    let shape = shape
    let rows = shape[safe: 1, 0]
    
    if axis == 0 {
      var row: [Tensor] = []
      for d in 0..<value.count {
        let feature = value[d]
        if row.isEmpty {
          row = feature.batched(into: into).map { Tensor($0) }
        } else {
          let current = feature.batched(into: into).map { Tensor($0) }
          row = zip(row, current).map { $0.0.concat($0.1, axis: 2) }
        }
      }
      return row
    } else if axis == 1 {
      var result: [Tensor] = []
      for d in 0..<value.count {
        var row: [Tensor] = []

        for r in 0..<rows {
          if row.isEmpty {
            let col = value[d][r].batched(into: into).map { Tensor($0) }
            row = col
          } else {
            let col = value[d][r].batched(into: into).map { Tensor($0) }
            row = zip(row, col).map { $0.0.concat($0.1, axis: 0)}
          }
        }
        
        if result.isEmpty {
          result = row
        } else {
          result = zip(result, row).map { $0.0.concat($0.1, axis: 2)}
        }
      }
      
      return result
    } else {
      return [self]
    }
  }
  
  func mean(axis: Int = -1) -> Tensor {
    let block: MathBlock = { feature in
      feature.average
    }
    
    if axis == -1 {
      let total = self.shape.reduce(1, *)
      let all = self.value.flatten().sum / Tensor.Scalar(total)
      return Tensor(all)
    }
    
    return apply(axis: axis, block)
  }
  
  func sum(axis: Int = -1) -> Tensor {
    if axis == -1 {
      return Tensor(value.sum)
    } else {
      return apply(axis: axis) { feature in
        feature.sum
      }
    }
  }
  
  func subtract(axis: Int = -1) -> Tensor {
    if axis == -1 {
      return Tensor(value.flatten().reduce(0, -))
    } else {
      return apply(axis: axis) { feature in
        var feature = feature
        let first = feature.first ?? 0
        feature = Array(feature.dropFirst())
        return feature.reduce(first, -)
      }
    }
  }
  
  func multiply(axis: Int = -1) -> Tensor {
    if axis == -1 {
      return Tensor(value.flatten().reduce(1, *))
    } else {
      return apply(axis: axis) { feature in
        feature.reduce(1, *)
      }
    }
  }
  
  func norm(axis: Int = -1) -> Tensor {
    let block: MathBlock = { feature in
      sqrt(feature.sumOfSquares)
    }
    
    if axis == -1 {
      return Tensor(sqrt(self.value.sumOfSquares))
    }
    
    return apply(axis: axis, block)
  }
  
  @discardableResult
  func concat(_ tensor: Tensor, axis: Int = 1) -> Tensor {
    let shape = shape
    let rows = shape[safe: 1, 0]
    let depth = shape[safe: 2, 0]
    
    var new = self.value
    
    if axis == -1 {
      var flatSelf = value.flatten()
      flatSelf.append(contentsOf: tensor.value.flatten())
      return Tensor(flatSelf, context: context)
    }
    
    if axis == 2 {
      new.append(contentsOf: tensor.value)
    } else {
      for d in 0..<depth {
        if axis == 0 {
          new[d].append(contentsOf: tensor.value[safe: d] ?? [])
        }
        for r in 0..<rows {
          if axis == 1 {
            new[d][r].append(contentsOf: tensor.value[safe: d]?[safe: r] ?? [])
          }
        }
      }
    }

    return Tensor(new, context: context)
  }
  
  func l2Normalize() {
    let flatValue: Tensor.Scalar = value.sumOfSquares
    let normalized = value / sqrt(flatValue)
    self.value = normalized
  }
  
  func map(_ transform: (Tensor.Scalar) -> Tensor.Scalar) -> Tensor {
    var result: Tensor.Data = []
    
    self.value.forEach { d in
      var row: [[Tensor.Scalar]] = []
      d.forEach { r in
        row.append(r.map(transform))
      }
      result.append(row)
    }
    
    return Tensor(result, context: context)
  }

  static func /(lhs: Scalar, rhs: Tensor) -> Tensor {
    let newTensorValue = lhs / rhs.value
    return Tensor(newTensorValue, context: rhs.context)
  }
  
  static func *(lhs: Scalar, rhs: Tensor) -> Tensor {
    let newTensorValue = lhs * rhs.value
    return Tensor(newTensorValue, context: rhs.context)
  }
  
  static func -(lhs: Scalar, rhs: Tensor) -> Tensor {
    let newTensorValue = lhs - rhs.value
    return Tensor(newTensorValue, context: rhs.context)
  }
  
  static func /(lhs: Tensor, rhs: Scalar) -> Tensor {
    let left = lhs.value
    
    let newTensorValue = left / rhs
    return Tensor(newTensorValue, context: lhs.context)
  }
  
  static func *(lhs: Tensor, rhs: Scalar) -> Tensor {
    let left = lhs.value
    
    let newTensorValue = left * rhs
    return Tensor(newTensorValue, context: lhs.context)
  }
  
  static func -(lhs: Tensor, rhs: Scalar) -> Tensor {
    let left = lhs.value
    
    let newTensorValue = left - rhs
    return Tensor(newTensorValue, context: lhs.context)
  }
  
  static func +(lhs: Tensor, rhs: Scalar) -> Tensor {
    let left = lhs.value
    
    let newTensorValue = left + rhs
    return Tensor(newTensorValue, context: lhs.context)
  }
  
  static func +(lhs: Tensor, rhs: Tensor) -> Tensor {
    let left = lhs.value
    let right = rhs.value
    
    let newTensor = left + right
    return Tensor(newTensor, context: lhs.context)
  }
  
  static func -(lhs: Tensor, rhs: Tensor) -> Tensor {
    let left = lhs.value
    let right = rhs.value
    
    let newTensor = left - right
    return Tensor(newTensor, context: lhs.context)
  }
  
  static func *(lhs: Tensor, rhs: Tensor) -> Tensor {
    let left = lhs.value
    let right = rhs.value
    
    let newTensor = left * right
    return Tensor(newTensor, context: lhs.context)
  }
  
  static func /(lhs: Tensor, rhs: Tensor) -> Tensor {
    let left = lhs.value
    let right = rhs.value
    
    let newTensor = left / right
    return Tensor(newTensor, context: lhs.context)
  }
  
  func zerosLike() -> Tensor {
    let shape = shape
    let rows = shape[safe: 1, 0]
    let columns = shape[safe: 0, 0]
    let depth = shape[safe: 2, 0]
    
    return Tensor(NumSwift.zerosLike((rows, columns, depth)))
  }

  func onesLike() -> Tensor {
    let shape = shape
    let rows = shape[safe: 1, 0]
    let columns = shape[safe: 0, 0]
    let depth = shape[safe: 2, 0]
    
    return Tensor(NumSwift.onesLike((rows, columns, depth)))
  }
}

extension Tensor: CustomDebugStringConvertible {
  public var debugDescription: String {
    var string = """
                 <Tensor \n
                 """
    
    let shape = shape
    string += "shape: (col: \(shape[safe: 0, 0]), rows: \(shape[safe: 1, 0]), depth: \(shape[safe: 2, 0]))\n"
    string += "-----\n"
    string += "value: \n"
    value.forEach { depth in
      depth.forEach { string += "\($0)\n" }
      string += "-----\n"
    }
    
    string += "graph: \(graph != nil)\n"
    string += ">"
    return string
  }
}

extension Array where Element == Tensor {
  
  var mean: Tensor {
    var mutableSelf = self
    let first = mutableSelf.removeFirst()
    let mean = mutableSelf.reduce(first, +) / count.asTensorScalar
    return mean
  }
  
  func gradients(_ deltas: [Tensor]) -> [Tensor.Gradient] {
    var result = [Tensor.Gradient](repeating: .init(),
                                      count: deltas.count)
    
    let workerCount = Swift.min(Constants.maxWorkers, Int(ceil(Double(deltas.count) / 4)))
    deltas.concurrentForEach(workers: workerCount) { element, index in
      let delta = deltas[index]
      let output = self[index]
      result[index] = output.gradients(delta: delta)
    }
    
    return result
  }
  
  static func +(lhs: [Tensor], rhs: [Tensor]) -> Self {
    var result: [Tensor] = []
    
    for i in 0..<lhs.count {
      let left = lhs[i]
      let right = rhs[i]
      result.append(left + right)
    }
    
    return result
  }
  
  static func -(lhs: [Tensor], rhs: [Tensor]) -> Self {
    var result: [Tensor] = []
    
    for i in 0..<lhs.count {
      let left = lhs[i]
      let right = rhs[i]
      result.append(left - right)
    }
    
    return result
  }
  
  static func *(lhs: [Tensor], rhs: [Tensor]) -> Self {
    var result: [Tensor] = []
    
    for i in 0..<lhs.count {
      let left = lhs[i]
      let right = rhs[i]
      result.append(left * right)
    }
    
    return result
  }
  
  static func /(lhs: [Tensor], rhs: [Tensor]) -> Self {
    var result: [Tensor] = []
    
    for i in 0..<lhs.count {
      let left = lhs[i]
      let right = rhs[i]
      result.append(left / right)
    }
    
    return result
  }
  
  static func *(lhs: [Tensor], rhs: Tensor.Scalar) -> Self {
    var result: [Tensor] = []
    
    for i in 0..<lhs.count {
      let left = lhs[i]
      result.append(left * rhs)
    }
    
    return result
  }
  
  static func -(lhs: [Tensor], rhs: Tensor.Scalar) -> Self {
    var result: [Tensor] = []
    
    for i in 0..<lhs.count {
      let left = lhs[i]
      result.append(left - rhs)
    }
    
    return result
  }
  
  static func /(lhs: [Tensor], rhs: Tensor.Scalar) -> Self {
    var result: [Tensor] = []
    
    for i in 0..<lhs.count {
      let left = lhs[i]
      result.append(left / rhs)
    }
    
    return result
  }
  
  static func +(lhs: [Tensor], rhs: Tensor.Scalar) -> Self {
    var result: [Tensor] = []
    
    for i in 0..<lhs.count {
      let left = lhs[i]
      result.append(left + rhs)
    }
    
    return result
  }
}

public extension Tensor.Gradient {
  static func applyMultiple(lhs: Tensor.Gradient,
                            rhs: Tensor.Gradient,
                            block: (_ lhs: [Tensor], _ rhs: [Tensor]) -> [Tensor]) -> Tensor.Gradient {
    //(input: [Tensor], weights: [Tensor], biases: [Tensor])
    let input = block(lhs.input, rhs.input)
    let weight = block(lhs.weights, rhs.weights)
    let bias = block(lhs.biases, rhs.biases)
    return Tensor.Gradient(input: input, weights: weight, biases: bias)
  }
  
  static func applyScalar(lhs: Tensor.Gradient,
                          rhs: Tensor.Scalar,
                          block: (_ lhs: [Tensor], _ rhs: Tensor.Scalar) -> [Tensor]) -> Tensor.Gradient {
    //(input: [Tensor], weights: [Tensor], biases: [Tensor])
    let input = block(lhs.input, rhs)
    let weight = block(lhs.weights, rhs)
    let bias = block(lhs.biases, rhs)
    return Tensor.Gradient(input: input, weights: weight, biases: bias)
  }
  
  
  static func /(lhs: Tensor.Gradient, rhs: Tensor.Gradient) -> Tensor.Gradient {
    applyMultiple(lhs: lhs, rhs: rhs) { lhs, rhs in
      lhs / rhs
    }
  }
  
  static func *(lhs: Tensor.Gradient, rhs: Tensor.Gradient) -> Tensor.Gradient {
    applyMultiple(lhs: lhs, rhs: rhs) { lhs, rhs in
      lhs * rhs
    }
  }
  
  static func -(lhs: Tensor.Gradient, rhs: Tensor.Gradient) -> Tensor.Gradient {
    applyMultiple(lhs: lhs, rhs: rhs) { lhs, rhs in
      lhs - rhs
    }
  }
  
  static func +(lhs: Tensor.Gradient, rhs: Tensor.Gradient) -> Tensor.Gradient {
    applyMultiple(lhs: lhs, rhs: rhs) { lhs, rhs in
      lhs + rhs
    }
  }
  
  static func +(lhs: Tensor.Gradient, rhs: Tensor.Scalar) -> Tensor.Gradient {
    applyScalar(lhs: lhs, rhs: rhs) { lhs, rhs in
      lhs + rhs
    }
  }
  
  static func -(lhs: Tensor.Gradient, rhs: Tensor.Scalar) -> Tensor.Gradient {
    applyScalar(lhs: lhs, rhs: rhs) { lhs, rhs in
      lhs - rhs
    }
  }
  
  static func /(lhs: Tensor.Gradient, rhs: Tensor.Scalar) -> Tensor.Gradient {
    applyScalar(lhs: lhs, rhs: rhs) { lhs, rhs in
      lhs / rhs
    }
  }
  
  static func *(lhs: Tensor.Gradient, rhs: Tensor.Scalar) -> Tensor.Gradient {
    applyScalar(lhs: lhs, rhs: rhs) { lhs, rhs in
      lhs * rhs
    }
  }
}
