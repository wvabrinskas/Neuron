//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/27/22.
//

import Foundation
import NumSwift

public extension Float {
  static var stabilityFactor: Self {
    1e-12
  }
}

#if arch(arm64)
public extension Float16 {
  static var stabilityFactor: Self {
    1e-4
  }
}
#endif

public extension Tensor {
  typealias MathBlock = (_ feature: [Scalar]) -> Scalar
  typealias MathAlongBlock = (_ feature: [Scalar], _ value: ([Scalar]?, Scalar?)) -> [Scalar]

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
      
    } else if axis == 1  {
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
  
  static func axisToApplyAlong(selfSize: TensorSize, size: TensorSize) -> Int? {
    if size.columns == selfSize.columns,
       size.rows == 1,
       size.depth == selfSize.depth {
      return 0
      
    } else if size.columns == 1,
              size.rows == selfSize.rows,
              size.depth == selfSize.depth {
      return 1
      
    } else if size.columns == selfSize.columns,
              size.rows == selfSize.rows,
              size.depth == 1 {
      return 2
    } else {
      return nil
    }
  }
  
  func applyAlong(axis: Int, input: Tensor, _ block: MathAlongBlock) -> Tensor {
    let shape = input.shape
    let size = TensorSize(array: shape)

    let selfShape = self.shape
    let selfSize = TensorSize(array: selfShape)
    
    var featureResults: [[[Scalar]]] = []
    
    for d in 0..<selfSize.depth {
      var result: [[Scalar]] = []
      
      for r in 0..<selfSize.rows {
        
        let location = TensorSize(rows: r, columns: 1, depth: d)
        let feature = value[d][r]
        
        let out: [Scalar]
        
        if axis == 0,
           size.columns == selfSize.columns,
           size.rows == 1,
           size.depth == selfSize.depth {
          
          let v = input.value[safe: location.depth, [[]]][safe: 0, []]
          out = block(feature, (v, nil))
          
        } else if axis == 1,
                  size.columns == 1,
                  size.rows == selfSize.rows,
                  size.depth == selfSize.depth {
          
          let v = input.value[safe: location.depth, [[]]][safe: location.rows, []][safe: 0, 0]
          out = block(feature, (nil, v))
          
        } else if axis == 2,
                  size.columns == selfSize.columns,
                  size.rows == selfSize.rows,
                  size.depth == 1 {
          
          let v = input.value[safe: 0, [[]]][safe: location.rows, []]
          out = block(feature, (v, nil))
        } else {
          out = feature
        }
        
        result.append(out)
      }
      
      featureResults.append(result)
    }
    return Tensor(featureResults)
  }
  
  func divideAlong(axis: Int, value: Tensor) -> Tensor {
    let block: MathAlongBlock = { feature, value in
      if let valueArray = value.0 {
        return feature / valueArray
      } else if let valueScalar = value.1 {
        return feature / valueScalar
      } else {
        return feature
      }
    }
    
    let context = TensorContext { inputs, gradient in
      return (gradient * (1 / value), Tensor(), Tensor())
    }
    
    let out = applyAlong(axis: axis, input: value, block).value
    
    let new = Tensor(out, context: context)

    new.setGraph(self)
    
    return new
  }
  
  func multiplyAlong(axis: Int, value: Tensor) -> Tensor {
    let block: MathAlongBlock = { feature, value in
      if let valueArray = value.0 {
        return feature * valueArray
      } else if let valueScalar = value.1 {
        return feature * valueScalar
      } else {
        return feature
      }
    }
    
    let context = TensorContext { inputs, gradient in
      return (gradient * value, Tensor(), Tensor())
    }
    
    let out = applyAlong(axis: axis, input: value, block).value
    
    let new = Tensor(out, context: context)

    new.setGraph(self)
    
    return new
  }
  
  func addAlong(axis: Int, value: Tensor) -> Tensor {
    let block: MathAlongBlock = { feature, value in
      if let valueArray = value.0 {
        return feature + valueArray
      } else if let valueScalar = value.1 {
        return feature + valueScalar
      } else {
        return feature
      }
    }
    
    let context = TensorContext { inputs, gradient in
      /// wrt to inputs it's 1
      return (gradient, Tensor(), Tensor())
    }
    
    let out = applyAlong(axis: axis, input: value, block).value
    
    let new = Tensor(out, context: context)

    new.setGraph(self)
    
    return new
  }
  
  func subtractAlong(axis: Int, value: Tensor) -> Tensor {
    let block: MathAlongBlock = { feature, value in
      if let valueArray = value.0 {
        return feature - valueArray
      } else if let valueScalar = value.1 {
        return feature - valueScalar
      } else {
        return feature
      }
    }
    
    let context = TensorContext { inputs, gradient in
      return (gradient * -1, Tensor(), Tensor())
    }
    
    let out = applyAlong(axis: axis, input: value, block).value
    
    let new = Tensor(out, context: context)

    new.setGraph(self)
    
    return new
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
  
  func sqrt(adding: Tensor.Scalar = .stabilityFactor) -> Tensor {
    let result: [[[Tensor.Scalar]]] = value.map { $0.map { $0.map { Tensor.Scalar.sqrt($0 + adding) } } }
    let tensor = Tensor(result, context: context)
    return tensor
  }
  
  func variance(axis: Int = -1) -> Tensor {
    let block: MathBlock = { feature in
      let mean = feature.mean
      let sumOSquares = (feature - mean).sumOfSquares
      
      let count = feature.count
            
      return sumOSquares / Tensor.Scalar(count)
    }
    
    if axis == -1 {
      let mean = self.mean(axis: -1).asScalar()
      let flat = self.value.flatten()
      let sumOfSquares = (flat - mean).sumOfSquares
      
      return Tensor(sumOfSquares / Tensor.Scalar(flat.count))
    }
    
    return apply(axis: axis, block)
  }
  
  func mean(axis: Int = -1) -> Tensor {
    let block: MathBlock = { feature in
      feature.mean
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
      Tensor.Scalar.sqrt(feature.sumOfSquares)
    }
    
    if axis == -1 {
      return Tensor(Tensor.Scalar.sqrt(self.value.sumOfSquares))
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
  
  func l2Normalized() -> Tensor {
    let flatValue: Tensor.Scalar = value.sumOfSquares
    let normalized = value / Tensor.Scalar.sqrt(flatValue)
    return Tensor(normalized, context: context)
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
    
    if let axis = Tensor.axisToApplyAlong(selfSize: TensorSize(array: lhs.shape),
                                          size: TensorSize(array: rhs.shape)) {
      return lhs.addAlong(axis: axis, value: rhs)
    }
    
    let newTensor = left + right
    
    let context = TensorContext { inputs, gradient in
      return (gradient, Tensor(), Tensor())
    }
    
    let new = Tensor(newTensor, context: context)
    
    new.setGraph(lhs)
    
    return new
  }
  
  static func -(lhs: Tensor, rhs: Tensor) -> Tensor {
    let left = lhs.value
    let right = rhs.value
    
    if let axis = Tensor.axisToApplyAlong(selfSize: TensorSize(array: lhs.shape),
                                          size: TensorSize(array: rhs.shape)) {
      return lhs.subtractAlong(axis: axis, value: rhs)
    }
    
    let newTensor = left - right
    
    let context = TensorContext { inputs, gradient in
      return (gradient * -1, Tensor(), Tensor())
    }
    
    let new = Tensor(newTensor, context: context)
    
    new.setGraph(lhs)
    
    return new
  }
  
  static func *(lhs: Tensor, rhs: Tensor) -> Tensor {
    let left = lhs.value
    let right = rhs.value
    
    if let axis = Tensor.axisToApplyAlong(selfSize: TensorSize(array: lhs.shape),
                                          size: TensorSize(array: rhs.shape)) {
      return lhs.multiplyAlong(axis: axis, value: rhs)
    }
    
    let newTensor = left * right
    
    let context = TensorContext { inputs, gradient in
      return (gradient * rhs, Tensor(), Tensor())
    }
    
    let new = Tensor(newTensor, context: context)
    
    new.setGraph(lhs)
    
    return new
  }
  
  static func /(lhs: Tensor, rhs: Tensor) -> Tensor {
    let left = lhs.value
    let right = rhs.value
    
    if let axis = Tensor.axisToApplyAlong(selfSize: TensorSize(array: lhs.shape),
                                          size: TensorSize(array: rhs.shape)) {
      return lhs.divideAlong(axis: axis, value: rhs)
    }
    
    let newTensor = left / right
    
    let context = TensorContext { inputs, gradient in
      return (gradient * (1 / rhs), Tensor(), Tensor())
    }
    
    let new = Tensor(newTensor, context: context)
    
    new.setGraph(lhs)
    
    return new
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
  
  func transposed() -> Tensor {
    Tensor(self.value.transpose2d(), context: context)
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
    string += "label: \(label)\n"
    string += "-----\n"
    string += "value: \n"
    value.forEach { depth in
      depth.forEach { string += "\($0)\n" }
      string += "-----\n"
    }
    
    string += "graph: \(graph.isEmpty == false)\n"
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
  
  func gradients(_ deltas: [Tensor], wrt: [Tensor]) -> [Tensor.Gradient] {
    var result = [Tensor.Gradient](repeating: .init(),
                                      count: deltas.count)
    
    let workerCount = Constants.maxWorkers
    deltas.concurrentForEach(workers: workerCount) { element, index in
      let delta = deltas[index]
      let input = wrt[index]
      let output = self[index]
      result[index] = output.gradients(delta: delta, wrt: input)
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

public extension Tensor {
 static func fillWith(value: Tensor.Scalar, size: TensorSize) -> Tensor {
    var result: [[[Tensor.Scalar]]]  = []
    
    for _ in 0..<size.depth {
      var row: [[Tensor.Scalar]] = []
      for _ in 0..<size.rows {
        row.append([Tensor.Scalar](repeating: value, count: size.columns))
      }
      result.append(row)
    }

    return Tensor(result)
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
