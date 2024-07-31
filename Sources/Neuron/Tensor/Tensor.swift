//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/26/22.
//

import Foundation
import NumSwift
import NumSwiftC
import Numerics

public typealias TensorNumeric = Real & Codable

public protocol TensorRange {
  associatedtype T: RangeExpression<Int>
  var range: T { get }
}

/// The fundamental base for all arithmetic in the network. It holds a reference to the backpropgation graph as well as the values of the forward pass.
/// Its `value` property is a 3D array for all instances.
public class Tensor<N: TensorNumeric>: Equatable, Codable {
  public static func == (lhs: Tensor<N>, rhs: Tensor<N>) -> Bool {
    lhs.value == rhs.value || lhs.id == rhs.id  // not sure at all why there's an ID property
  }

  public typealias Scalar = N
  public typealias Data = [[[Scalar]]]
  
  /// Gradient object returned from `gradient` calculation on the Tensor<N>. Contains gradients w.r.t to the `input`, w.r.t to the `weights`, and w.r.t to the `biases`
  public struct Gradient {
    let input: [Tensor<Scalar>]
    let weights: [Tensor<Scalar>]
    let biases: [Tensor<Scalar>]
    
    public init(input: [Tensor<Scalar>] = [],
                weights: [Tensor<Scalar>] = [],
                biases: [Tensor<Scalar>] = []) {
      self.input = input
      self.weights = weights
      self.biases = biases
    }
  }
  
  /// Description label
  public var label: String = ""
  
  /// Generic id
  public var id: UUID = UUID()
  
  /// Actual numerical value of the Tensor<N>
  public var value: Data {
    didSet {
      shapeCache = value.shape
    }
  }
  
  /// Flattens the `value` and returns if there is any content in the array.
  public var isEmpty: Bool {
    shape == [0,0,0]
  }
  
  internal var graph: Tensor<Scalar>?
  internal let context: TensorContext<Scalar>
  
  /// Shape of the Tensor<N> as a 1D array. `[columns, rows, depth]`
  public var shape: [Int] {
    guard shapeCache.isEmpty else { return shapeCache }
    let s = value.shape
    shapeCache = s
    return s
  }
  
  /// Input from the graph
  public var input: Tensor<Scalar> {
    graph ?? Tensor<Scalar>()
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
                   _ depthRange: some RangeExpression<Int>) -> Tensor<Scalar> {
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
        
    return Tensor<Scalar>(data, context: context)
  }
  

  /// Default initializer with no context or value
  public init() {
    self.value = []
    self.context = TensorContext<Scalar>()
  }
  
  /// Initializer for Tensor<N> with a scalar value
  /// - Parameters:
  ///   - data: `[[Scalar]]` object to set
  ///   - context: Backpropagation context
  public init(_ data: Scalar? = nil, context: TensorContext<Scalar> = TensorContext<Scalar>()) {
    if let data = data {
      self.value = [[[data]]]
    } else {
      self.value = []
    }
    
    self.context = context
  }
  
  /// Initializer for Tensor<N> with a fully 1D array
  /// - Parameters:
  ///   - data: `[Scalar]` object to set
  ///   - context: Backpropagation context
  public init(_ data: [Scalar], context: TensorContext<Scalar> = TensorContext<Scalar>()) {
    self.value = [[data]]
    self.context = context
  }
  
  /// Initializer for Tensor<N> with a fully 2D array
  /// - Parameters:
  ///   - data: `[[Scalar]]` object to set
  ///   - context: Backpropagation context
  public init(_ data: [[Scalar]], context: TensorContext<Scalar> = TensorContext<Scalar>()) {
    self.value = [data]
    self.context = context
  }
  
  /// Initializer for Tensor<N> with a fully 3D array
  /// - Parameters:
  ///   - data: `Tensor<N>.Data` object to set
  ///   - context: Backpropagation context
  public init(_ data: Data, context: TensorContext<Scalar> = TensorContext<Scalar>()) {
    self.value = data
    self.context = context
  }
  
  /// Prints the current graph all the way to the input.
  public func printGraph() {
    var t: Tensor<N>? = self
    
    while let g = t {
      print("value: ", g.value, "input: ", g.input.value)
      t = g.graph
    }
  }
  
  /// Checks if the value of the Tensor<N> is the same as another Tensor<N>. `==` checks id property.
  /// - Parameter to: Tensor<N> to compare to
  /// - Returns: Bool indicating if the values are equal
  public func isValueEqual(to: Tensor<Scalar>) -> Bool {
    self.value == to.value
  }
  
  /// Sets the input graph to this Tensor<N>
  /// - Parameter tensor: The tensor to insert into the graph
  public func setGraph(_ tensor: Tensor<Scalar>) {
    self.graph = tensor
  }
  
  /// Calculates the gradients in the Tensor<N> graph
  /// - Parameter delta: The gradient to backpropagate w.r.t
  /// - Returns: A Gradient where the the `inputGradients` is the gradient w.r.t each input in the graph at each layer and `weightGradients` is the gradient w.r.t to each parameter at each layer.
  public func gradients(delta: Tensor<Scalar>) -> Tensor<Scalar>.Gradient {
    var inputGradients: [Tensor<Scalar>] = []
    var weightGradients: [Tensor<Scalar>] = []
    var biasGradients: [Tensor<Scalar>] = []

    var tensor: Tensor<Scalar>? = self
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
  
  /// Remove this Tensor<N> from the graph.
  /// - Returns: Detached Tensor<N>
  public func detached() -> Tensor<Scalar> {
    Tensor<Scalar>(value, context: TensorContext<Scalar>())
  }
  
  /// Gets the `Tensor<N>.Scalar` value of this Tensors value. This is reserved for Tensor's that have a value of size `[1, 1, 1]` aka a `Scalar` as `[[[Scalar]]]`
  /// - Returns: The scalar value.
  public func asScalar() -> Scalar {
    value[safe: 0, [[]]][safe: 0, []][safe: 0, 0]
  }
}
