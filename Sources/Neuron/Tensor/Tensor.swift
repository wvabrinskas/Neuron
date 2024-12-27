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
  public var value: Data {
    didSet {
      shapeCache = value.shape
    }
  }
  
  /// Flattens the `value` and returns if there is any content in the array.
  public var isEmpty: Bool {
    shape == [0,0,0]
  }
  
  internal var graph: [UUID: Tensor] = [:]
  internal let context: TensorContext
  
  /// Shape of the Tensor as a 1D array. `[columns, rows, depth]`
  public var shape: [Int] {
    guard shapeCache.isEmpty else { return shapeCache }
    let s = value.shape
    shapeCache = s
    return s
  }
  
  /// Input from the graph
  public var input: [UUID: Tensor] {
    graph
  }
  
  // cache the shape so we dont need to calculate it each time we call for shape
  private var shapeCache: [Int] = []
 
  enum CodingKeys: String, CodingKey {
    case label
    case id
    case context
    case value
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
    
    self.context = context
  }
  
  /// Initializer for Tensor with a fully 1D array
  /// - Parameters:
  ///   - data: `[Scalar]` object to set
  ///   - context: Backpropagation context
  public init(_ data: [Scalar], context: TensorContext = TensorContext()) {
    self.value = [[data]]
    self.context = context
  }
  
  /// Initializer for Tensor with a fully 2D array
  /// - Parameters:
  ///   - data: `[[Scalar]]` object to set
  ///   - context: Backpropagation context
  public init(_ data: [[Scalar]], context: TensorContext = TensorContext()) {
    self.value = [data]
    self.context = context
  }
  
  /// Initializer for Tensor with a fully 3D array
  /// - Parameters:
  ///   - data: `Tensor.Data` object to set
  ///   - context: Backpropagation context
  public init(_ data: Data, context: TensorContext = TensorContext()) {
    self.value = data
    self.context = context
  }
  
  /// Prints the current graph all the way to the input.
  public func printGraph() {
    var inputs: [UUID: Tensor] = input
    
    // print self
    // print children
    // repeat for children
    print("base: \(id) \(shape) \(label)")
    var level = 0
    while inputs.isEmpty == false {
      var i = 0
      var childrenAtLevel: [UUID: Tensor] = [:]
      print("root \(level) ------ ")
      for (k, v) in inputs {
        print("     input: \(i) \(k): \(v.shape) \(v.label)")//print("input \(k): ", v)
        childrenAtLevel.merge(v.input) { _, new in
          new
        }
        i += 1
      }
      
      level += 1
      inputs = childrenAtLevel
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
    self.graph[tensor.id] = tensor
  }
  
  /// Calculates the gradients in the Tensor graph
  /// - Parameter delta: The gradient to backpropagate w.r.t
  /// - Returns: A Gradient where the the `inputGradients` is the gradient w.r.t each input in the graph at each layer and `weightGradients` is the gradient w.r.t to each parameter at each layer.
  public func gradients(delta: Tensor) -> Tensor.Gradient {
    
    let selfGradients = getGradients(delta: delta)
    var inputGradients: [Tensor] = selfGradients.input
    var weightGradients: [Tensor] = selfGradients.weight
    var biasGradients: [Tensor] = selfGradients.bias
    
    var gradientsAtLevelToUse: [Tensor] = inputGradients
    var childrenAtLevelToUse: [UUID: Tensor] = input

    while childrenAtLevelToUse.isEmpty == false {
      var gradientsAtLevel: [Tensor] = []
      var childrenAtLevel: [UUID: Tensor] = [:]
      
      var i = 0

      for input in childrenAtLevelToUse.values {
        let gradientToUse = gradientsAtLevelToUse[i]
      
        let newGrads = input.getGradients(delta: gradientToUse)
        
        inputGradients.insert(contentsOf: newGrads.input, at: 0)
        weightGradients.insert(contentsOf: newGrads.weight, at: 0)
        biasGradients.insert(contentsOf: newGrads.bias, at: 0)
        
        i += 1

        gradientsAtLevel.append(contentsOf: newGrads.input)
        input.input.forEach { childrenAtLevel[$0] = $1 }
      }
      
      gradientsAtLevelToUse = gradientsAtLevel
      childrenAtLevelToUse = childrenAtLevel
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
  
  func getGradients(delta: Tensor) -> (input: [Tensor], weight: [Tensor], bias: [Tensor]) {
    var inputGradients: [Tensor] = []
    var weightGradients: [Tensor] = []
    var biasGradients: [Tensor] = []
    
    // backpropogate self
    for input in graph.values {
      let newGrads = context.backpropagate(input, delta)

      inputGradients.insert(newGrads.input, at: 0)
      weightGradients.insert(newGrads.weight, at: 0)
      biasGradients.insert(newGrads.bias, at: 0)
    }
    
    return (inputGradients, weightGradients, biasGradients)
  }
}
