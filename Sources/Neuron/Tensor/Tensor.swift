//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/26/22.
//

import Foundation
import NumSwift
import NumSwiftC

public protocol TensorRange {
  associatedtype T: RangeExpression<Int>
  var range: T { get }
}

/// The fundamental base for all arithmetic in the network. It holds a reference to the backpropgation graph as well as the values of the forward pass.
/// Its `value` property is a 3D array for all instances.
public class Tensor: Equatable, Codable {
  public static func == (lhs: Tensor, rhs: Tensor) -> Bool {
    lhs.value == rhs.value || lhs.id == rhs.id  // not sure at all why there's an ID property
  }

  #if QUANTIZED_F16
  public typealias Scalar = Float16
  #else
  public typealias Scalar = Float
  #endif
  
  public typealias Data = [[[Scalar]]]
  
  /// Gradient object returned from `gradient` calculation on the Tensor. Contains gradients w.r.t to the `input`, w.r.t to the `weights`, and w.r.t to the `biases`
  public struct Gradient {
    let input: [Tensor]
    let weights: [Tensor]
    let biases: [Tensor]
    
    public init(input: [Tensor] = [],
                weights: [Tensor] = [],
                biases: [Tensor] = []) {
      self.input = input
      self.weights = weights
      self.biases = biases
    }
  }
  
  /// Description label
  public var label: String = ""
  
  /// Generic id
  public var id: UUID = UUID()
  
  /// Actual numerical value of the Tensor
  public private(set) var value: Data {
    didSet {
      shapeCache = value.shape
    }
  } 
  
  /// Flattens the `value` and returns if there is any content in the array.
  public var isEmpty: Bool {
    shape == [0,0,0]
  }
  
  internal var graph: Tensor?
  internal let context: TensorContext
  
  /// Shape of the Tensor as a 1D array. `[columns, rows, depth]`
  public var shape: [Int] {
    guard shapeCache.isEmpty else { return shapeCache }
    let s = value.shape
    shapeCache = s
    return s
  }
  
  /// Input from the graph
  public var input: Tensor {
    graph ?? Tensor()
  }
  
  /// Hack to avoid having to rewrite every single math function that revolves around 1d and 2d arrays.
  /// This returns the number of features of a given tensor determined by the specific shape of the array.
  /// Ideally we'd use `depth` for this, however that requires a lot of rewrite around arithmetic functions.
  /// In itit I tried to change `1d` to `3d` but for loop each element and appending `[[element]]`.
  /// Very similar in `2D` as well where I appended `[element]` instead.
  public var features: Int = 1
  
  // cache the shape so we dont need to calculate it each time we call for shape
  private var shapeCache: [Int] = []
 
  enum CodingKeys: String, CodingKey {
    case label
    case id
    case context
    case value
  }
  
  public subscript(feature index: Int) -> [[Tensor.Scalar]] {
    let size = TensorSize(array: shape)
    
    if size.depth == features {
      return value[index]
    } else if size.rows == features {
      return [value[0][index]]
    } else {
      return [[value[0][0][index]]]
    }
  }
  
  /// only works for 3D tensors, Input is [colRange, rowRange, depthRange]
  public subscript(_ colRange: some RangeExpression<Int>,
                   _ rowRange: some RangeExpression<Int>,
                   _ depthRange: some RangeExpression<Int>) -> Tensor {
    var data: Data = []

    for d in depthRange.relative(to: self.value) {
      var rows: [[Scalar]] = []
      for r in rowRange.relative(to: self.value[d]) {
        var row: [Scalar] = []
        for c in colRange.relative(to: self.value[d][r]) {
          row.append(self.value[d][r][c])
        }
        rows.append(row)
      }
      data.append(rows)
    }
        
    return Tensor(data, context: context)
  }
  

  /// Default initializer with no context or value
  public init() {
    self.value = []
    self.context = TensorContext()
  }
  
  /// Initializer for Tensor with a scalar value
  /// - Parameters:
  ///   - data: `[[Scalar]]` object to set
  ///   - context: Backpropagation context
  public init(_ data: Scalar? = nil, context: TensorContext = TensorContext()) {
    if let data = data {
      self.value = [[[data]]]
    } else {
      self.value = []
    }
    
    self.features = 1
    self.context = context
  }
  
  /// Initializer for Tensor with a fully 1D array
  /// - Parameters:
  ///   - data: `[Scalar]` object to set
  ///   - context: Backpropagation context
  public init(_ data: [Scalar], context: TensorContext = TensorContext()) {
    self.value = []
    
    for d in data {
      self.value.append([[d]])
    }
    
    self.context = context
    self.features = data.count
  }
  
  /// Initializer for Tensor with a fully 2D array
  /// - Parameters:
  ///   - data: `[[Scalar]]` object to set
  ///   - context: Backpropagation context
  public init(_ data: [[Scalar]], context: TensorContext = TensorContext()) {
    self.value = []
    
    for d in data {
      self.value.append([d])
    }
    
    self.context = context
    self.features = data.count
  }
  
  /// Initializer for Tensor with a fully 3D array
  /// - Parameters:
  ///   - data: `Tensor.Data` object to set
  ///   - context: Backpropagation context
  public init(_ data: Data, context: TensorContext = TensorContext()) {
    self.value = data
    self.context = context
    self.features = data.count
  }
  
  /// Prints the current graph all the way to the input.
  public func printGraph() {
    var t: Tensor? = self
    
    while let g = t {
      print("value: ", g.value, "input: ", g.input.value)
      t = g.graph
    }
  }
  
  /// Checks if the value of the Tensor is the same as another Tensor. `==` checks id property.
  /// - Parameter to: Tensor to compare to
  /// - Returns: Bool indicating if the values are equal
  public func isValueEqual(to: Tensor) -> Bool {
    self.value == to.value
  }
  
  /// Sets the input graph to this Tensor
  /// - Parameter tensor: The tensor to insert into the graph
  public func setGraph(_ tensor: Tensor) {
    self.graph = tensor
  }
  
  /// Calculates the gradients in the Tensor graph
  /// - Parameter delta: The gradient to backpropagate w.r.t
  /// - Returns: A Gradient where the the `inputGradients` is the gradient w.r.t each input in the graph at each layer and `weightGradients` is the gradient w.r.t to each parameter at each layer.
  public func gradients(delta: Tensor) -> Tensor.Gradient {
    var inputGradients: [Tensor] = []
    var weightGradients: [Tensor] = []
    var biasGradients: [Tensor] = []

    var tensor: Tensor? = self
    var incomingGradient = delta

    while let tensorNode = tensor {

      if let input = tensorNode.graph {
        let newGrads = tensorNode.context.backpropagate(input, incomingGradient)
        incomingGradient = newGrads.input
        
        inputGradients.insert(newGrads.input, at: 0)
        weightGradients.insert(newGrads.weight, at: 0)
        biasGradients.insert(newGrads.bias, at: 0)
      }
            
      tensor = tensorNode.graph
    }

    return .init(input: inputGradients, weights: weightGradients, biases: biasGradients)
  }
  
  /// Remove this Tensor from the graph.
  /// - Returns: Detached Tensor
  public func detached() -> Tensor {
    Tensor(value, context: TensorContext())
  }
  
  /// Gets the `Tensor.Scalar` value of this Tensors value. This is reserved for Tensor's that have a value of size `[1, 1, 1]` aka a `Scalar` as `[[[Scalar]]]`
  /// - Returns: The scalar value.
  public func asScalar() -> Scalar {
    value[safe: 0, [[]]][safe: 0, []][safe: 0, 0]
  }
  
  public func l2Normalize() {
    let flatValue: Tensor.Scalar = value.sumOfSquares
    let normalized = value / Tensor.Scalar.sqrt(flatValue)
    self.value = normalized
  }
  
  public func clip(_ val: Scalar = 0.01) {
    value = value.map { $0.map { $0.map { Swift.max(-val, Swift.min(val, $0)) }}}
  }
}
