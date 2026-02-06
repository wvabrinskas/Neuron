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
  public typealias ID = UInt64
  
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
  public private(set) var id: ID = .defaultValue()
  
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
  
  internal var graphChain: Set<ID> = []
  internal var graph: [ID: Tensor] = [:]
  internal let context: TensorContext
  
  /// Shape of the Tensor as a 1D array. `[columns, rows, depth]`
  public var shape: [Int] {
    guard shapeCache.isEmpty else { return shapeCache }
    let s = value.shape
    shapeCache = s
    return s
  }
  
  /// Input from the graph
  public var input: [ID: Tensor] {
    graph
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
  
  required public init(from decoder: any Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.label = try container.decode(String.self, forKey: .label)
    self.context = try container.decode(TensorContext.self, forKey: .context)
    self.value = try container.decode(Tensor.Data.self, forKey: .value)
    
    // support old models with UUIDs or Int64 (if we back out of using Int64)
    if let id = try? container.decodeIfPresent(Tensor.ID.self, forKey: .id) {
      self.id = id
    } else {
      self.id = .defaultValue()
    }
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
    setId()
  }
  
  /// Initializer for Tensor with a fully 1D array
  /// - Parameters:
  ///   - data: `[Scalar]` object to set
  ///   - context: Backpropagation context
  public init(_ data: [Scalar], context: TensorContext = TensorContext()) {
    self.value = [[data]]
    self.context = context
    self.features = data.count
    setId()
  }
  
  /// Initializer for Tensor with a fully 2D array
  /// - Parameters:
  ///   - data: `[[Scalar]]` object to set
  ///   - context: Backpropagation context
  public init(_ data: [[Scalar]], context: TensorContext = TensorContext()) {
    self.value = [data]
    self.context = context
    self.features = data.count
    setId()
  }
  
  /// Initializer for Tensor with a fully 3D array
  /// - Parameters:
  ///   - data: `Tensor.Data` object to set
  ///   - context: Backpropagation context
  public init(_ data: Data, context: TensorContext = TensorContext()) {
    self.value = data
    self.context = context
    self.features = data.count
    setId()
  }
  
  private func setId() {
    self.id = IDGenerator.shared.explicitInt64()
  }
  
  /// Prints the current graph all the way to the input.
  public func printGraph(wrt: Tensor? = nil, deep: Bool = false) {
    var inputs: [ID: Tensor] = input
    
    if let wrt {
      if graphChain.contains(wrt.id) == false {
        print("no connection")
        return
      }
    }

    
    // print self
    // print children
    // repeat for children
    var outputString: [String] = []
    
    outputString.insert("output: \(id) \(shape) \(label) \n", at: 0)
    
    var level = 0
    while inputs.isEmpty == false {
      var i = 0
      var childrenAtLevel: [ID: Tensor] = [:]

      for (k, v) in inputs {
        childrenAtLevel.merge(v.input) { _, new in
          new
        }
        
        if let wrt {
          if v.graphChain.contains(wrt.id) {
            outputString.insert("     branch: \(i) \(k): \(v.shape) \(v.label) \n", at: 0)//print("input \(k): ", v)
          }
          
          if wrt.id == v.id {
            print(outputString.joined())
            return
          }
        } else {
          outputString.insert("     branch: \(i) \(k): \(v.shape) \(v.label) \n", at: 0)//print("input \(k): ", v)
        }
        
        i += 1
      }
      
      outputString.insert("level: \(level) ------ \n", at: 0)
      

      level += 1
      if let wrt, childrenAtLevel.count > 1 {
        inputs = childrenAtLevel.filter({ $0.value.graphChain.contains(wrt.id) || $0.value.graphChain.isEmpty })
      } else {
        inputs = childrenAtLevel
      }
      
      if deep {
        inputs.forEach { _, v in
          v.printGraph(deep: deep)
        }
      }

    }
    
    outputString.append("""
                        \t\t\t|
                        \t\t\t|
                        \t\t\tV
                        """)
    
    print(outputString.joined())
  }
  
  /// Checks if the value of the Tensor is the same as another Tensor. `==` checks id property.
  /// - Parameter to: Tensor to compare to
  /// - Returns: Bool indicating if the values are equal
  public func isValueEqual(to: Tensor) -> Bool {
    self.value == to.value
  }
  
  /// Checks if the value of the Tensor is the same as another Tensor. `==` checks id property.
  /// - Parameter to: Tensor to compare to
  /// - Returns: Bool indicating if the values are equal
  public func isValueEqual(to: Tensor, accuracy: Tensor.Scalar = 0.000001) -> Bool {
    guard shape == to.shape else { return false }
    
    for (lhs, rhs) in zip(self.value, to.value) {
      for (lhsElement, rhsElement) in zip(lhs, rhs) {
        for (lhsScalar, rhsScalar) in zip(lhsElement, rhsElement) {
          if abs(lhsScalar - rhsScalar) > accuracy {
            return false
          }
        }
      }
    }
    return true
  }
  
  /// Sets the input graph to this Tensor
  /// - Parameter tensor: The tensor to insert into the graph
  /// - Parameter breakCycles: If true, will create a detached copy of the tensor to prevent reference cycles (default: false)
  public func setGraph(_ tensor: Tensor, breakCycles: Bool = false) {
    let tensorToStore = breakCycles ? tensor.detached() : tensor
    graph[tensorToStore.id] = tensorToStore
    graphChain.insert(tensorToStore.id)
    graphChain.formUnion(tensorToStore.graphChain)
  }
  
  /// Sets the input graph with cycle detection - if the tensor already exists in the graph chain, it will be detached
  /// - Parameter tensor: The tensor to insert into the graph
  internal func setGraphSafe(_ tensor: Tensor) {
    // If this tensor is already in our chain or if we're in its chain, break the cycle
    let shouldBreakCycle = graphChain.contains(tensor.id) || tensor.graphChain.contains(self.id) || tensor.id == self.id
    setGraph(tensor, breakCycles: shouldBreakCycle)
  }
  
  /// Calculates the gradients in the Tensor graph
  /// - Parameter delta: The gradient to backpropagate w.r.t
  /// - Parameter wrt: Optional parameter to tell the auto grad which input Tensor in the graph to backprop to, this is inclusive of the wrt tensor. That tensor's gradients will be calculated as well wrt to its input. If this isn't provided it will return all inputs at every level of the graph in a single array.
  /// - Returns: A Gradient where the the `inputGradients` is the gradient w.r.t each input in the graph at each layer and `weightGradients` is the gradient w.r.t to each parameter at each layer.
  public func gradients(delta: Tensor, wrt: Tensor? = nil) -> Tensor.Gradient {
    
    let selfGradients = getGradients(delta: delta, wrt: wrt)
    var inputGradients: [Tensor] = selfGradients.input
    var weightGradients: [Tensor] = selfGradients.weight
    var biasGradients: [Tensor] = selfGradients.bias
    
    var gradientsAtLevelToUse: [Tensor] = inputGradients
    var childrenAtLevelToUse: [ID: Tensor] = input
    
    if let wrt {
      childrenAtLevelToUse = childrenAtLevelToUse.filter({ $0.value.graphChain.contains(wrt.id) || $0.value.id == wrt.id })
    }
    
    func process(input: Tensor,
                 wrt: Tensor? = nil,
                 gradientToUse: Tensor,
                 childrenAtLevel: inout [ID: Tensor],
                 gradientsAtLevel: inout [Tensor]) {
      let newGrads = input.getGradients(delta: gradientToUse, wrt: wrt)
      
      inputGradients.insert(contentsOf: newGrads.input, at: 0)
      weightGradients.insert(contentsOf: newGrads.weight, at: 0)
      biasGradients.insert(contentsOf: newGrads.bias, at: 0)
      
      gradientsAtLevel.append(contentsOf: newGrads.input)
      input.input.forEach { childrenAtLevel[$0] = $1 }
    }

    while childrenAtLevelToUse.isEmpty == false {
      var gradientsAtLevel: [Tensor] = []
      var childrenAtLevel: [ID: Tensor] = [:]
      
      for (i, input) in childrenAtLevelToUse.values.enumerated() {
                                        
        let gradientToUse = gradientsAtLevelToUse[i]
        
        if let wrt {
          
          // only process the gradients for an input that actually dealt with the wrt tensor
          if input.graphChain.contains(wrt.id) {
            process(input: input,
                    wrt: wrt,
                    gradientToUse: gradientToUse,
                    childrenAtLevel: &childrenAtLevel,
                    gradientsAtLevel: &gradientsAtLevel)
          }
          
          if wrt.id == input.id {
            return .init(input: inputGradients, weights: weightGradients, biases: biasGradients)
          }
        } else {
          process(input: input,
                  gradientToUse: gradientToUse,
                  childrenAtLevel: &childrenAtLevel,
                  gradientsAtLevel: &gradientsAtLevel)
        }

      }
      
      gradientsAtLevelToUse = gradientsAtLevel
      if let wrt, childrenAtLevel.count > 1 {
        // check if on the right chain or at the right node
        childrenAtLevelToUse = childrenAtLevel.filter({ $0.value.graphChain.contains(wrt.id) || $0.value.id == wrt.id })
      } else {
        childrenAtLevelToUse = childrenAtLevel
      }
    }
    
    return .init(input: inputGradients, weights: weightGradients, biases: biasGradients)
  }
  
  /// Remove this Tensor from the graph.
  /// - Returns: Detached Tensor
  public func detached() -> Tensor {
    let tensor = Tensor(value, context: TensorContext())
    tensor.id = self.id
    return tensor
  }
  
  /// Remove this Tensor from the graph, copies the value, changes the ID, and optionally removes or keeps the graph context.
  /// - Returns: Copied Tensor
  public func copy(keepContext: Bool = false) -> Tensor {
    guard keepContext == false else {
      return Tensor(value, context: context)
    }
    
    return Tensor(value)
  }
  
  public func isScalar() -> Bool {
    shape == [1,1,1]
  }
  
  /// Gets the `Tensor.Scalar` value of this Tensors value. This is reserved for Tensor's that have a value of size `[1, 1, 1]` aka a `Scalar` as `[[[Scalar]]]`
  /// - Returns: The scalar value.
  public func asScalar() -> Scalar {
    value[safe: 0, [[]]][safe: 0, []][safe: 0, 0]
  }
  
  func getGradients(delta: Tensor, wrt: Tensor? = nil) -> (input: [Tensor], weight: [Tensor], bias: [Tensor]) {
    var inputGradients: [Tensor] = []
    var weightGradients: [Tensor] = []
    var biasGradients: [Tensor] = []
    
    // backpropogate self
    for input in graph.values {
      if input.id != wrt?.id {
        if let wrt, input.graphChain.contains(wrt.id) == false {
          continue
        }
      }
      
      let newGrads = context.backpropagate(input, delta, wrt)

      inputGradients.insert(newGrads.input, at: 0)
      weightGradients.insert(newGrads.weight, at: 0)
      biasGradients.insert(newGrads.bias, at: 0)
    }
    
    return (inputGradients, weightGradients, biasGradients)
  }
  
  public func l2Normalize() {
    let flatValue: Tensor.Scalar = value.sumOfSquares
    let normalized = value / Tensor.Scalar.sqrt(flatValue + Tensor.Scalar.stabilityFactor)
    self.value = normalized
  }
  
  public func l2Norm() -> Scalar {
    Tensor.Scalar.sqrt(value.sumOfSquares)
  }
  
  public func clip(_ val: Scalar = 0.01) {
    value = value.map { $0.map { $0.map { Swift.max(-val, Swift.min(val, $0)) }}}
  }
}
