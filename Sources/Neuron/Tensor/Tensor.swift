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

  public var label: String = ""
  public var id: UUID = UUID()
  public var value: Data
  public var isEmpty: Bool {
    value.flatten().isEmpty
  }
  
  internal var graph: Tensor?
  internal let context: TensorContext
  
  public var shape: [Int] {
    value.shape
  }
  
  public var input: Tensor {
    graph ?? Tensor()
  }
  
  public init() {
    self.value = []
    self.context = TensorContext()
  }
  
  public init(_ data: Scalar? = nil, context: TensorContext = TensorContext()) {
    if let data = data {
      self.value = [[[data]]]
    } else {
      self.value = []
    }
    
    self.context = context
  }
  
  public init(_ data: [Scalar], context: TensorContext = TensorContext()) {
    self.value = [[data]]
    self.context = context
  }
  
  public init(_ data: [[Scalar]], context: TensorContext = TensorContext()) {
    self.value = [data]
    self.context = context
  }
  
  public init(_ data: Data, context: TensorContext = TensorContext()) {
    self.value = data
    self.context = context
  }
  
  public func printGraph() {
    var t: Tensor? = self
    
    while let g = t {
      print("value: ", g.value, "input: ", g.input.value)
      t = g.graph
    }
  }
  
  public func isValueEqual(to: Tensor) -> Bool {
    self.value == to.value
  }
  
  public func setGraph(_ tensor: Tensor) -> Tensor {
    tensor.graph = self
    return tensor
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
  
  public func detached() -> Tensor {
    Tensor(value, context: TensorContext())
  }
  
  public func asScalar() -> Scalar {
    value[safe: 0, [[]]][safe: 0, []][safe: 0, 0]
  }
}
