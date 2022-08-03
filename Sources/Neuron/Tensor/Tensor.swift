//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/26/22.
//

import Foundation
import NumSwift
import NumSwiftC

public class Tensor: Equatable, Codable {
  public static func == (lhs: Tensor, rhs: Tensor) -> Bool {
    lhs.id == rhs.id
  }
  
  public typealias Scalar = Float
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
  public var value: Data
  
  /// Flattens the `value` and returns if there is any content in the array.
  public var isEmpty: Bool {
    value.flatten().isEmpty
  }
  
  internal var graph: Tensor?
  internal let context: TensorContext
  
  /// Shape of the Tensor as a 1D array. `[columns, rows, depth]`
  public var shape: [Int] {
    value.shape
  }
  
  /// Input from the graph
  public var input: Tensor {
    graph ?? Tensor()
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

    var tensor: Tensor? = self
    var incomingGradient = delta
    
    var currentBiasGrads: [Scalar] = []
    incomingGradient.value.forEach { val in
      currentBiasGrads.append(val.sum)
    }
  
    var biasGradients: [Tensor] = [Tensor(currentBiasGrads)]

    while let tensorNode = tensor {
      
      if let input = tensorNode.graph {
        let newGrads = tensorNode.context.backpropagate(input, incomingGradient)
        incomingGradient = newGrads.input
        
        inputGradients.insert(newGrads.input, at: 0)
        weightGradients.insert(newGrads.weight, at: 0)
        
        // automatically calculates bias gradients
        var currentBiasGrads: [Scalar] = []
        newGrads.input.value.forEach { val in
          currentBiasGrads.append(val.sum)
        }
        
        biasGradients.insert(Tensor(currentBiasGrads), at: 0)
      }
            
      tensor = tensorNode.graph
    }

    // there is no bias for the input layer
    biasGradients.removeFirst()
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
}
