//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/26/22.
//

import Foundation
import NumSwift
import NumSwiftC

/// A protocol that wraps a `RangeExpression<Int>` for use in tensor subscript operations.
/// Conforming types provide a `range` property used to slice tensor dimensions.
public protocol TensorRange {
  associatedtype T: RangeExpression<Int>
  var range: T { get }
}


/// The fundamental base for all arithmetic in the network. It holds a reference to the backpropgation graph as well as the values of the forward pass.
/// Internally stores data as a flat `Tensor.Value` with `TensorSize` metadata.
/// The `value` computed property provides backward-compatible access as a 3D `[[[Scalar]]]` array.
public class Tensor: Equatable, Codable {
  /// Compares two tensors by storage/shape or identity.
  ///
  /// Equality returns `true` when values and sizes match, or when both tensors
  /// share the same internal identifier.
  ///
  /// - Parameters:
  ///   - lhs: Left-hand tensor.
  ///   - rhs: Right-hand tensor.
  /// - Returns: `true` when tensors represent the same value or identity.
  public static func == (lhs: Tensor, rhs: Tensor) -> Bool {
    lhs.storage == rhs.storage && lhs.size == rhs.size || lhs.id == rhs.id
  }

  #if QUANTIZED_F16
/// A type alias representing an individual scalar element stored in a `Tensor`.
  /// Resolves to `Float16` when the `Float16` compiler flag is set, otherwise `Float`.
  public typealias Scalar = Float16
  #else
/// A type alias representing an individual scalar element stored in a `Tensor`.
  /// Resolves to `Float` when the `Float16` compiler flag is not set.
  public typealias Scalar = Float
  #endif
  
  /// The legacy nested-array type for tensor data. Prefer using `storage` and `size` for new code.
  public typealias Data = [[[Scalar]]]
/// A flat contiguous array of `Scalar` values used as the internal storage for tensor data.
  public typealias Value = ContiguousArray<Scalar> //[Scalar]
/// A unique identifier type for `Tensor` instances.
  public typealias ID = UInt64
  
  /// Gradient object returned from `gradient` calculation on the Tensor. Contains gradients w.r.t to the `input`, w.r.t to the `weights`, and w.r.t to the `biases`
  public struct Gradient {
    let input: [Tensor]
    let weights: [Tensor]
    let biases: [Tensor]
    
    /// Creates a gradient container for input, weight, and bias derivatives.
    ///
    /// - Parameters:
    ///   - input: Gradients with respect to upstream inputs.
    ///   - weights: Gradients with respect to trainable weights.
    ///   - biases: Gradients with respect to trainable biases.
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
  
  // MARK: - Flat Storage
  
  /// The flat contiguous storage for this tensor's data.
  /// Memory layout: `index = d * rows * columns + r * columns + c`
  /// 
  public internal(set) var storage: Tensor.Value
  
  /// The shape metadata (columns, rows, depth)
  public internal(set) var size: TensorSize
  
  /// Flattens the `value` and returns if there is any content in the array.
  public var isEmpty: Bool {
    storage.isEmpty || shape == [0, 0, 0]
  }
  
  internal var graphChain: Set<ID> = []
  internal var graph: [ID: Tensor] = [:]
  internal let context: TensorContext
  
  /// Shape of the Tensor as a 1D array. `[columns, rows, depth]`
  public var shape: [Int] {
    size.asArray
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
  
  enum CodingKeys: String, CodingKey {
    case label
    case id
    case context
    case value
    case storage
    case size
  }
  
  // MARK: - Flat Indexing Helpers
  
  /// Computes the flat index for a given (column, row, depth) coordinate.
  /// Memory layout: `index = d * rows * columns + r * columns + c`
  @inline(__always)
/// Computes the flat storage index for an element at the given (column, row, depth) coordinates.
  ///
  /// - Parameters:
  ///   - c: The column index.
  ///   - r: The row index.
  ///   - d: The depth index.
  /// - Returns: The corresponding flat index into the `storage` array.
  public func flatIndex(column c: Int, row r: Int, depth d: Int) -> Int {
    d * size.rows * size.columns + r * size.columns + c
  }
  
  /// Subscript for direct element access using (column, row, depth) coordinates.
  public subscript(c: Int, r: Int, d: Int) -> Scalar {
    get { storage[flatIndex(column: c, row: r, depth: d)] }
    set { storage[flatIndex(column: c, row: r, depth: d)] = newValue }
  }
  
  /// only works for 3D tensors, Input is [colRange, rowRange, depthRange]
  public subscript(_ colRange: some RangeExpression<Int>,
                   _ rowRange: some RangeExpression<Int>,
                   _ depthRange: some RangeExpression<Int>) -> Tensor {
    let depthBound = 0..<size.depth
    let rowBound = 0..<size.rows
    let colBound = 0..<size.columns
    
    let dRange = depthRange.relative(to: depthBound)
    let rRange = rowRange.relative(to: rowBound)
    let cRange = colRange.relative(to: colBound)
    
    let newDepth = dRange.count
    let newRows = rRange.count
    let newCols = cRange.count
    
    var result = Tensor.Value(repeating: 0, count: newDepth * newRows * newCols)
    let newSize = TensorSize(rows: newRows, columns: newCols, depth: newDepth)
    
    var idx = 0
    for d in dRange {
      for r in rRange {
        for c in cRange {
          result[idx] = storage[flatIndex(column: c, row: r, depth: d)]
          idx += 1
        }
      }
    }
    
    return Tensor(result, size: newSize, context: context)
  }
  
  // MARK: - Initializers

  /// Default initializer with no context or value
  public init() {
    self.storage = Tensor.Value()
    self.size = TensorSize(rows: 0, columns: 0, depth: 0)
    self.context = TensorContext()
  }

  required public init(from decoder: any Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.label = try container.decode(String.self, forKey: .label)
    self.context = try container.decode(TensorContext.self, forKey: .context)
    
    // Decode from nested array format for backward compatibility
    if let nestedValue = try container.decodeIfPresent(Tensor.Data.self, forKey: .value) {
      let depth = nestedValue.count
      var maxRows = 0
      var maxCols = 0
      for depthSlice in nestedValue {
        maxRows = Swift.max(maxRows, depthSlice.count)
        for row in depthSlice {
          maxCols = Swift.max(maxCols, row.count)
        }
      }
      self.size = TensorSize(rows: maxRows, columns: maxCols, depth: depth)
      
      let totalCount = maxCols * maxRows * depth
      var flat = Tensor.Value(repeating: 0, count: totalCount)
      for d in 0..<depth {
        let depthSlice = nestedValue[d]
        for r in 0..<depthSlice.count {
          let row = depthSlice[r]
          let baseIndex = d * maxRows * maxCols + r * maxCols
          for c in 0..<row.count {
            flat[baseIndex + c] = row[c]
          }
        }
      }
      self.storage = flat
    } else {
      let storage = try container.decode(Tensor.Value.self, forKey: .storage)
      let size = try container.decode(TensorSize.self, forKey: .size)
      
      self.storage = storage
      self.size = size
    }
    
    // support old models with UUIDs or Int64 (if we back out of using Int64)
    if let id = try? container.decodeIfPresent(Tensor.ID.self, forKey: .id) {
      self.id = id
    } else {
      self.id = .defaultValue()
    }
  }
  
  /// Initializer for Tensor with a scalar value
  /// - Parameters:
  ///   - data: `Scalar` object to set
  ///   - context: Backpropagation context
  public init(_ data: Scalar? = nil, context: TensorContext = TensorContext()) {
    if let data = data {
      self.storage = Tensor.Value([data])
      self.size = TensorSize(rows: 1, columns: 1, depth: 1)
    } else {
      self.storage = Tensor.Value()
      self.size = TensorSize(rows: 0, columns: 0, depth: 0)
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
    // 1D data is stored as shape (data.count, 1, 1) -- matching the old [[data]] layout
    self.storage = Tensor.Value(data)
    self.size = TensorSize(rows: 1, columns: data.count, depth: 1)
    self.context = context
    self.features = data.count
    setId()
  }
  
  /// Initializer for Tensor with a fully 2D array. Expecting every component to have the same number of columns
  /// - Parameters:
  ///   - data: `[[Scalar]]` object to set
  ///   - context: Backpropagation context
  public init(_ data: [[Scalar]], context: TensorContext = TensorContext()) {
    // 2D data is stored as shape (cols, rows, 1) -- matching the old [data] layout
    let rows = data.count
    let cols = data[safe: 0]?.count ?? 0
    
    self.size = TensorSize(rows: rows, columns: cols, depth: 1)
    
    let totalCount = cols * rows
    var flat = Tensor.Value(repeating: 0, count: totalCount)
    for r in 0..<rows {
      let row = data[r]
      let baseIndex = r * cols
      for c in 0..<row.count {
        flat[baseIndex + c] = row[c]
      }
    }
    
    self.storage = flat
    self.context = context
    self.features = data.count
    setId()
  }
  
  /// Initializer for Tensor with a fully 3D array, Expecting every component to have the same number of rows and columns
  /// - Parameters:
  ///   - data: `Tensor.Data` object to set
  ///   - context: Backpropagation context
  public init(_ data: Data, context: TensorContext = TensorContext()) {
    let depth = data.count
    let rows = data[safe: 0]?.count ?? 0
    let columns = data[safe: 0]?[safe: 0]?.count ?? 0
    
    self.size = TensorSize(rows: rows, columns: columns, depth: depth)
    
    let totalCount = columns * rows * depth
    var flat = Tensor.Value(repeating: 0, count: totalCount)
    
    // Fill the flat array, zero-padding any ragged edges
    for d in 0..<depth {
      let depthSlice = data[d]
      for r in 0..<depthSlice.count {
        let row = depthSlice[r]
        let baseIndex = d * rows * columns + r * columns
        for c in 0..<row.count {
          flat[baseIndex + c] = row[c]
        }
      }
    }
    
    self.storage = flat
    self.context = context
    self.features = data.count
    setId()
  }
  
  /// Direct initializer from flat storage and size. No copy is performed.
  /// - Parameters:
  ///   - storage: Flat contiguous array of scalar values
  ///   - size: Shape metadata (columns, rows, depth)
  ///   - context: Backpropagation context
  public init(_ storage: Tensor.Value, size: TensorSize, context: TensorContext = TensorContext()) {
    self.storage = storage
    self.size = size
    self.context = context
    self.features = size.depth
    setId()
  }

      
  /// Extracts one depth slice as a flat row-major `Tensor.Value` directly from storage.
  /// Each slice has `rows * columns` elements.
  /// - Parameter d: The depth index (0-based)
  /// - Returns: A flat contiguous array of the depth slice
  public func depthSlice(_ d: Int) -> Tensor.Value {
    let sliceSize = size.rows * size.columns
    let start = d * sliceSize
    return Tensor.Value(storage[start..<(start + sliceSize)])
  }
  
  /// MARK: - Depth Slice Access
  /// 
  /// Writes a flat depth slice back into storage.
  /// - Parameters:
  ///   - d: The depth index (0-based)
  ///   - data: The flat row-major data to write (must have rows * columns elements)
  public func setDepthSlice(_ d: Int, _ data: Tensor.Value) {
    let sliceSize = size.rows * size.columns
    let start = d * sliceSize
    for i in 0..<sliceSize {
      storage[start + i] = data[i]
    }
  }
  
  /// Creates a new Tensor from a single depth slice of this tensor.
  /// The result has depth=1 and the same rows/columns.
  public func depthSliceTensor(_ d: Int) -> Tensor {
    let sliceStorage = depthSlice(d)
    return Tensor(sliceStorage, size: TensorSize(rows: size.rows, columns: size.columns, depth: 1))
  }
  
  private func setId() {
    self.id = IDGenerator.shared.explicitUInt64()
  }
  
  // MARK: - Nested Array Reconstruction
  
  /// Reconstructs the 3D nested array `[[[Scalar]]]` from flat storage and size metadata.
  /// This is used by the `value` computed property for backward compatibility.
  internal func toNestedArray() -> Data {
    guard !storage.isEmpty else { return [] }
    
    let columns = size.columns
    let rows = size.rows
    let depth = size.depth
    
    let expectedCount = columns * rows * depth
    precondition(storage.count == expectedCount,
                 "Tensor storage/size mismatch: storage has \(storage.count) elements but size \(size) expects \(expectedCount)")
    
    var result: Data = []
    result.reserveCapacity(depth)
    
    for d in 0..<depth {
      var depthSlice: [[Scalar]] = []
      depthSlice.reserveCapacity(rows)
      
      for r in 0..<rows {
        let start = d * rows * columns + r * columns
        let end = start + columns
        depthSlice.append(Array(storage[start..<end]))
      }
      result.append(depthSlice)
    }
    
    return result
  }
  
  // MARK: - Graph
  
  /// Prints a human-readable view of this tensor's computation graph.
  ///
  /// - Parameters:
  ///   - wrt: Optional node to constrain printed branches to a specific path.
  ///   - deep: When `true`, recursively prints each child node's graph.
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
  
  // MARK: - Value Comparison
  
  /// Checks if the value of the Tensor is the same as another Tensor. `==` checks id property.
  /// - Parameter to: Tensor to compare to
  /// - Returns: Bool indicating if the values are equal
  public func isValueEqual(to: Tensor) -> Bool {
    self.storage == to.storage && self.size == to.size
  }
  
  /// Checks if the value of the Tensor is the same as another Tensor. `==` checks id property.
  /// - Parameter to: Tensor to compare to
  /// - Parameter accuracy: The accuracy to use for the comparison
  /// - Returns: Bool indicating if the values are equal
  public func isValueEqual(to: Tensor, accuracy: Tensor.Scalar = 0.000001) -> Bool {
    guard size == to.size else { return false }
    
    for i in 0..<storage.count {
      if abs(storage[i] - to.storage[i]) > accuracy {
        return false
      }
    }
    return true
  }
  
  // MARK: - Graph Management
  
  /// Sets the input graph to this Tensor
  /// - Parameter tensor: The tensor to insert into the graph
  /// - Parameter breakCycles: If true, will create a copy of the tensor to prevent reference cycles, keeping the context of the original tensor (default: false)
  public func setGraph(_ tensor: Tensor, breakCycles: Bool = false) {
    let tensorToStore = breakCycles ? tensor.copy(keepContext: true) : tensor
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
  
  // MARK: - Copy / Detach
  
  /// Remove this Tensor from the graph.
  /// - Returns: Detached Tensor
  public func detached() -> Tensor {
    let tensor = Tensor(storage, size: size, context: TensorContext())
    tensor.id = self.id
    return tensor
  }
  
  /// Remove this Tensor from the graph, copies the value, changes the ID, and optionally removes or keeps the graph context.
  /// - Returns: Copied Tensor
  public func copy(keepContext: Bool = false) -> Tensor {
    guard keepContext == false else {
      return Tensor(storage, size: size, context: context)
    }
    
    return Tensor(storage, size: size)
  }
  
  /// Indicates whether this tensor represents exactly one scalar value.
  ///
  /// - Returns: `true` when shape is `[1, 1, 1]`.
  public func isScalar() -> Bool {
    size == TensorSize(rows: 1, columns: 1, depth: 1)
  }
  
  /// Gets the `Tensor.Scalar` value of this Tensors value. This is reserved for Tensor's that have a value of size `[1, 1, 1]` aka a `Scalar` as `[[[Scalar]]]`
  /// - Returns: The scalar value.
  public func asScalar() -> Scalar {
    storage.isEmpty ? 0 : storage[0]
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
  
  // MARK: - Normalization / Clipping
  
  /// Normalizes tensor values to unit L2 norm in place.
  public func l2Normalize() {
    let flatArray = Array(storage)
    let flatValue: Tensor.Scalar = flatArray.sumOfSquares
    let normalized = flatArray / Tensor.Scalar.sqrt(flatValue + Tensor.Scalar.stabilityFactor)
    self.storage = Tensor.Value(normalized)
  }
  
  /// Computes the L2 norm of the tensor values.
  ///
  /// - Returns: Euclidean norm over all scalar values.
  public func l2Norm() -> Scalar {
    Tensor.Scalar.sqrt(Array(storage).sumOfSquares)
  }
  
  /// Clamps every element into the symmetric range `[-val, val]`.
  ///
  /// - Parameter val: Absolute clamp bound.
  public func clip(_ val: Scalar = 0.01) {
    for i in 0..<storage.count {
      storage[i] = Swift.max(-val, Swift.min(val, storage[i]))
    }
  }
  
  // MARK: - Codable
  
  /// Encodes tensor metadata and flat storage for persistence.
  ///
  /// - Parameter encoder: Encoder used to serialize this tensor.
  public func encode(to encoder: any Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(label, forKey: .label)
    try container.encode(context, forKey: .context)
    // we no longer encode value as in the future we want to deprecate this. Tensor.Value in storage property is the properly stored data
    try container.encode(storage, forKey: .storage)
    try container.encode(size, forKey: .size)
    try container.encode(id, forKey: .id)
  }
  
}

// MARK: - Debug Description

extension Tensor: CustomDebugStringConvertible {
/// A human-readable description of the tensor, including its shape, label, and formatted values.
  /// Useful for debugging; prints each depth slice with row and column layout.
  public var debugDescription: String {
    var string = """
                 <Tensor \n
                 """
    
    string += "shape: (col: \(size.columns), rows: \(size.rows), depth: \(size.depth))\n"
    string += "-----\n"
    string += "label: \(label)\n"
    string += "-----\n"
    string += "value: \n"
    
    for d in 0..<size.depth {
      for r in 0..<size.rows {
        let start = d * size.rows * size.columns + r * size.columns
        let end = start + size.columns
        let row = Array(storage[start..<end])
        string += "\(row)\n"
      }
      string += "-----\n"
    }
    
    string += "graph: \(graph.isEmpty == false)\n"
    string += ">"
    return string
  }
}

// MARK: - Array<Tensor> Extensions

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

// MARK: - Gradient Extensions

public extension Tensor.Gradient {
  
  func l2NomalizeWeightsAndBiases() { 
    weights.forEach { $0.l2Normalize() }
    biases.forEach { $0.l2Normalize() }
  }
  
  func gradientL2NormClip(_ value: Tensor.Scalar = 1.0) -> Tensor.Gradient {
    let allWeights = weights.reduce(Tensor()) { partialResult, new in
      partialResult.concat(new, axis: 2)
    }
    let allBiases = biases.reduce(Tensor()) { partialResult, new in
      partialResult.concat(new, axis: 2)
    }

    // Single global norm across weights AND biases combined
    let weightNormSq = allWeights.l2Norm()
    let biasNormSq = allBiases.l2Norm()
    let globalNorm = Tensor.Scalar(sqrt(weightNormSq * weightNormSq + biasNormSq * biasNormSq))

    guard globalNorm > value else {
      return .init(input: input, weights: weights, biases: biases)
    }

    // One scaling factor applied to everything
    let scalingFactor = value / globalNorm

    return .init(
      input: input,
      weights: weights * scalingFactor,
      biases: biases * scalingFactor
    )
  }
  
  static func applyMultiple(lhs: Tensor.Gradient,
                            rhs: Tensor.Gradient,
                            block: (_ lhs: [Tensor], _ rhs: [Tensor]) -> [Tensor]) -> Tensor.Gradient {
    let input = block(lhs.input, rhs.input)
    let weight = block(lhs.weights, rhs.weights)
    let bias = block(lhs.biases, rhs.biases)
    return Tensor.Gradient(input: input, weights: weight, biases: bias)
  }
  
  static func applyScalar(lhs: Tensor.Gradient,
                          rhs: Tensor.Scalar,
                          block: (_ lhs: [Tensor], _ rhs: Tensor.Scalar) -> [Tensor]) -> Tensor.Gradient {
    let input = block(lhs.input, rhs)
    let weight = block(lhs.weights, rhs)
    let bias = block(lhs.biases, rhs)
    return Tensor.Gradient(input: input, weights: weight, biases: bias)
  }
  
  static func /(lhs: Tensor.Gradient, rhs: Tensor.Gradient) -> Tensor.Gradient {
    applyMultiple(lhs: lhs, rhs: rhs) { lhs, rhs in lhs / rhs }
  }
  
  static func *(lhs: Tensor.Gradient, rhs: Tensor.Gradient) -> Tensor.Gradient {
    applyMultiple(lhs: lhs, rhs: rhs) { lhs, rhs in lhs * rhs }
  }
  
  static func -(lhs: Tensor.Gradient, rhs: Tensor.Gradient) -> Tensor.Gradient {
    applyMultiple(lhs: lhs, rhs: rhs) { lhs, rhs in lhs - rhs }
  }
  
  static func +(lhs: Tensor.Gradient, rhs: Tensor.Gradient) -> Tensor.Gradient {
    applyMultiple(lhs: lhs, rhs: rhs) { lhs, rhs in lhs + rhs }
  }
  
  static func +(lhs: Tensor.Gradient, rhs: Tensor.Scalar) -> Tensor.Gradient {
    applyScalar(lhs: lhs, rhs: rhs) { lhs, rhs in lhs + rhs }
  }
  
  static func -(lhs: Tensor.Gradient, rhs: Tensor.Scalar) -> Tensor.Gradient {
    applyScalar(lhs: lhs, rhs: rhs) { lhs, rhs in lhs - rhs }
  }
  
  static func /(lhs: Tensor.Gradient, rhs: Tensor.Scalar) -> Tensor.Gradient {
    applyScalar(lhs: lhs, rhs: rhs) { lhs, rhs in lhs / rhs }
  }
  
  static func *(lhs: Tensor.Gradient, rhs: Tensor.Scalar) -> Tensor.Gradient {
    applyScalar(lhs: lhs, rhs: rhs) { lhs, rhs in lhs * rhs }
  }
}

// MARK: - Static Factory Methods

public extension Tensor {
  static func fillRandom(in range: ClosedRange<Tensor.Scalar> = 0...1, size: TensorSize) -> Tensor {
    let count = size.columns * size.rows * size.depth
    var storage = Tensor.Value(repeating: 0, count: count)
    
    for i in 0..<count {
      storage[i] = Tensor.Scalar.random(in: range)
    }
    
    return Tensor(storage, size: size)
  }
  
  static func fillWith(value: Tensor.Scalar, size: TensorSize) -> Tensor {
    let count = size.columns * size.rows * size.depth
    let storage = Tensor.Value(repeating: value, count: count)
    return Tensor(storage, size: size)
  }
}

