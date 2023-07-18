//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/27/22.
//

import Foundation
import NumSwift

public extension Tensor {
  typealias MathBlock = (_ feature: [[Scalar]]) -> Scalar
  
  func apply(axis: Int, _ block: MathBlock) -> Tensor {
    let shape = self.shape
    let rows = shape[safe: 1, 0]
    let columns = shape[safe: 0, 0]
    
    var result: [[[Scalar]]] = []
    
    if axis == 0 {
      var featureResults: [[Scalar]] = []
      
      for feature in self.value {
        var resultRow: [Scalar] = []
        for x in 0..<columns {
          var workingRow: [Scalar] = []

          for y in 0..<rows {
            workingRow.append(feature[y][x])
          }
            
          resultRow.append(block([workingRow]))
        }
        featureResults.append(resultRow)
      }

      result.append(featureResults)
      
    } else if axis == 1 {
      var featureResults: [[Scalar]] = []
      
      for feature in self.value {
        var resultRow: [Scalar] = []
        for y in 0..<columns {
          resultRow.append(block([feature[y]]))
        }
        featureResults.append(resultRow)
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
  
  func sum(axis: Int = -1) -> Tensor {
    let shape = shape
    let rows = shape[safe: 1, 0]
    let depth = shape[safe: 2, 0]
    let columns = shape[safe: 0, 0]
    
    if axis == 2 {
      return Tensor(value.sum)
    } else if axis == 1 || axis == -1 {
      var fullDepthResult: [[Float]] = []
      var depthResult: [Float] = []
      for d in 0..<depth {
        if axis == 1 {
          depthResult.append(value[d].sum)
        }
        var rowResult: [Float] = []
        for r in 0..<rows {
          if axis == -1 {
            rowResult.append(value[d][r].sum)
          }
        }

        fullDepthResult.append(rowResult)
      }
      
      if axis == 1 {
        return Tensor(depthResult, context: context)
      } else {
        return Tensor(fullDepthResult, context: context)
      }
    } else if axis == 0 {
      
      var result: [[Scalar]] = []
      for d in 0..<depth {
        if result.isEmpty {
          result = value[d]
        } else {
          result = result + value[d]
        }
      }
      
      return Tensor(result, context: context)
      
    } else {
      return self
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
  func concat(_ tensor: Tensor, axis: Int = -1) -> Tensor {
    let shape = shape
    let rows = shape[safe: 1, 0]
    let depth = shape[safe: 2, 0]
    
    var new = self.value
    
    if axis == 2 {
      new.append(contentsOf: tensor.value)
    } else {
      for d in 0..<depth {
        if axis == 1 {
          new[d].append(contentsOf: tensor.value[safe: d] ?? [])
        }
        for r in 0..<rows {
          if axis == -1 {
            new[d][r].append(contentsOf: tensor.value[safe: d]?[safe: r] ?? [])
          }
        }
      }
    }

    return Tensor(new, context: context)
  }
  
  func l2Normalize() {
    let flatValue: Float = value.sumOfSquares
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
    
    let workerCount = Int(ceil(Double(deltas.count) / 4))
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
