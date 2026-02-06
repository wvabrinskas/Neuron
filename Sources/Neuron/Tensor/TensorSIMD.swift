//
//  TensorSIMD.swift
//  Neuron
//
//  SIMD-optimized operations for small tensors (depth ≤ 4)
//  These are faster than NumSwiftC for tiny operations due to reduced overhead
//
//  Created by Claude Code
//

import Foundation
import simd
import Numerics

#if arch(arm64) || arch(x86_64)

// MARK: - SIMD Strategy

/// Determines if SIMD is beneficial for a given tensor operation
internal struct SIMDStrategy {
  /// Threshold for using SIMD vs NumSwiftC
  /// Below this element count, SIMD overhead is lower than NumSwiftC call
  static let elementThreshold = 256
  
  /// Returns true if SIMD should be used for this shape
  static func shouldUseSIMD(shape: [Int]) -> Bool {
    guard shape.count == 3 else { return false }
    let depth = shape[2]
    let totalElements = shape.reduce(1, *)
    
    // Use SIMD for:
    // 1. Small depths (RGB, RGBA)
    // 2. Small total element counts
    return depth <= 4 || totalElements < elementThreshold
  }
  
  /// Returns true if depth exactly matches a SIMD width
  static func hasNativeSIMDWidth(depth: Int) -> Bool {
    return depth == 2 || depth == 3 || depth == 4 || depth == 8 || depth == 16
  }
}

// MARK: - SIMD Operations for Depth=3 (RGB Images)

extension Tensor {
  
  /// Fast element-wise addition for depth-3 tensors using SIMD3
  internal func addSIMD3(_ rhs: Tensor) -> Tensor {
    precondition(shape[2] == 3 && rhs.shape[2] == 3, "SIMD3 requires depth=3")
    
    let rows = shape[1]
    let cols = shape[0]
    var result: [[[Scalar]]] = []
    
    for d in 0..<3 {
      var depthResult: [[Scalar]] = []
      
      for r in 0..<rows {
        var rowResult: [Scalar] = []
        rowResult.reserveCapacity(cols)
        
        // Process in SIMD3 chunks
        var c = 0
        while c + 2 < cols {
          // Load 3 elements as SIMD3
          let lhsVec = SIMD3<Scalar>(
            Scalar(value[d][r][c]),
            Scalar(value[d][r][c+1]),
            Scalar(value[d][r][c+2])
          )
          let rhsVec = SIMD3<Scalar>(
            Scalar(rhs.value[d][r][c]),
            Scalar(rhs.value[d][r][c+1]),
            Scalar(rhs.value[d][r][c+2])
          )
          
          let sum = lhsVec + rhsVec
          
          rowResult.append(Scalar(sum.x))
          rowResult.append(Scalar(sum.y))
          rowResult.append(Scalar(sum.z))
          
          c += 3
        }
        
        // Handle remainder
        while c < cols {
          rowResult.append(value[d][r][c] + rhs.value[d][r][c])
          c += 1
        }
        
        depthResult.append(rowResult)
      }
      result.append(depthResult)
    }
    
    return Tensor(result, context: context)
  }
  
  /// Fast element-wise subtraction for depth-3 tensors using SIMD3
  internal func subtractSIMD3(_ rhs: Tensor) -> Tensor {
    precondition(shape[2] == 3 && rhs.shape[2] == 3, "SIMD3 requires depth=3")
    
    let rows = shape[1]
    let cols = shape[0]
    var result: [[[Scalar]]] = []
    
    for d in 0..<3 {
      var depthResult: [[Scalar]] = []
      
      for r in 0..<rows {
        var rowResult: [Scalar] = []
        rowResult.reserveCapacity(cols)
        
        // Process in SIMD3 chunks
        var c = 0
        while c + 2 < cols {
          let lhsVec = SIMD3<Scalar>(
            Scalar(value[d][r][c]),
            Scalar(value[d][r][c+1]),
            Scalar(value[d][r][c+2])
          )
          let rhsVec = SIMD3<Scalar>(
            Scalar(rhs.value[d][r][c]),
            Scalar(rhs.value[d][r][c+1]),
            Scalar(rhs.value[d][r][c+2])
          )
          
          let difference = lhsVec - rhsVec
          
          rowResult.append(Scalar(difference.x))
          rowResult.append(Scalar(difference.y))
          rowResult.append(Scalar(difference.z))
          
          c += 3
        }
        
        // Handle remainder
        while c < cols {
          rowResult.append(value[d][r][c] - rhs.value[d][r][c])
          c += 1
        }
        
        depthResult.append(rowResult)
      }
      result.append(depthResult)
    }
    
    return Tensor(result, context: context)
  }
  
  /// Fast element-wise multiplication for depth-3 tensors using SIMD3
  internal func multiplySIMD3(_ rhs: Tensor) -> Tensor {
    precondition(shape[2] == 3 && rhs.shape[2] == 3, "SIMD3 requires depth=3")
    
    let rows = shape[1]
    let cols = shape[0]
    var result: [[[Scalar]]] = []
    
    for d in 0..<3 {
      var depthResult: [[Scalar]] = []
      
      for r in 0..<rows {
        var rowResult: [Scalar] = []
        rowResult.reserveCapacity(cols)
        
        var c = 0
        while c + 2 < cols {
          let lhsVec = SIMD3<Scalar>(
            Scalar(value[d][r][c]),
            Scalar(value[d][r][c+1]),
            Scalar(value[d][r][c+2])
          )
          let rhsVec = SIMD3<Scalar>(
            Scalar(rhs.value[d][r][c]),
            Scalar(rhs.value[d][r][c+1]),
            Scalar(rhs.value[d][r][c+2])
          )
          
          let product = lhsVec * rhsVec
          
          rowResult.append(Scalar(product.x))
          rowResult.append(Scalar(product.y))
          rowResult.append(Scalar(product.z))
          
          c += 3
        }
        
        while c < cols {
          rowResult.append(value[d][r][c] * rhs.value[d][r][c])
          c += 1
        }
        
        depthResult.append(rowResult)
      }
      result.append(depthResult)
    }
    
    return Tensor(result, context: context)
  }
  
  /// Fast element-wise division for depth-3 tensors using SIMD3
  internal func divideSIMD3(_ rhs: Tensor) -> Tensor {
    precondition(shape[2] == 3 && rhs.shape[2] == 3, "SIMD3 requires depth=3")
    
    let rows = shape[1]
    let cols = shape[0]
    var result: [[[Scalar]]] = []
    
    for d in 0..<3 {
      var depthResult: [[Scalar]] = []
      
      for r in 0..<rows {
        var rowResult: [Scalar] = []
        rowResult.reserveCapacity(cols)
        
        var c = 0
        while c + 2 < cols {
          let lhsVec = SIMD3<Scalar>(
            Scalar(value[d][r][c]),
            Scalar(value[d][r][c+1]),
            Scalar(value[d][r][c+2])
          )
          let rhsVec = SIMD3<Scalar>(
            Scalar(rhs.value[d][r][c]),
            Scalar(rhs.value[d][r][c+1]),
            Scalar(rhs.value[d][r][c+2])
          )
          
          let quotient = lhsVec / rhsVec
          
          rowResult.append(Scalar(quotient.x))
          rowResult.append(Scalar(quotient.y))
          rowResult.append(Scalar(quotient.z))
          
          c += 3
        }
        
        while c < cols {
          rowResult.append(value[d][r][c] / rhs.value[d][r][c])
          c += 1
        }
        
        depthResult.append(rowResult)
      }
      result.append(depthResult)
    }
    
    return Tensor(result, context: context)
  }
  
  /// Fast scalar addition for depth-3 tensors
  internal func addSIMD3Scalar(_ scalar: Scalar) -> Tensor {
    precondition(shape[2] == 3, "SIMD3 requires depth=3")
    
    let rows = shape[1]
    let cols = shape[0]
    var result: [[[Scalar]]] = []
    
    let scalarVec = SIMD3<Scalar>(repeating: Scalar(scalar))
    
    for d in 0..<3 {
      var depthResult: [[Scalar]] = []
      
      for r in 0..<rows {
        var rowResult: [Scalar] = []
        rowResult.reserveCapacity(cols)
        
        var c = 0
        while c + 2 < cols {
          let vec = SIMD3<Scalar>(
            Scalar(value[d][r][c]),
            Scalar(value[d][r][c+1]),
            Scalar(value[d][r][c+2])
          )
          
          let sum = vec + scalarVec
          
          rowResult.append(Scalar(sum.x))
          rowResult.append(Scalar(sum.y))
          rowResult.append(Scalar(sum.z))
          
          c += 3
        }
        
        while c < cols {
          rowResult.append(value[d][r][c] + scalar)
          c += 1
        }
        
        depthResult.append(rowResult)
      }
      result.append(depthResult)
    }
    
    return Tensor(result, context: context)
  }
  
  /// Fast scalar subtraction for depth-3 tensors
  internal func subtractSIMD3Scalar(_ scalar: Scalar) -> Tensor {
    precondition(shape[2] == 3, "SIMD3 requires depth=3")
    
    let rows = shape[1]
    let cols = shape[0]
    var result: [[[Scalar]]] = []
    
    let scalarVec = SIMD3<Scalar>(repeating: Scalar(scalar))
    
    for d in 0..<3 {
      var depthResult: [[Scalar]] = []
      
      for r in 0..<rows {
        var rowResult: [Scalar] = []
        rowResult.reserveCapacity(cols)
        
        var c = 0
        while c + 2 < cols {
          let vec = SIMD3<Scalar>(
            Scalar(value[d][r][c]),
            Scalar(value[d][r][c+1]),
            Scalar(value[d][r][c+2])
          )
          
          let difference = vec - scalarVec
          
          rowResult.append(Scalar(difference.x))
          rowResult.append(Scalar(difference.y))
          rowResult.append(Scalar(difference.z))
          
          c += 3
        }
        
        while c < cols {
          rowResult.append(value[d][r][c] - scalar)
          c += 1
        }
        
        depthResult.append(rowResult)
      }
      result.append(depthResult)
    }
    
    return Tensor(result, context: context)
  }
  
  /// Fast scalar multiplication for depth-3 tensors
  internal func multiplySIMD3Scalar(_ scalar: Scalar) -> Tensor {
    precondition(shape[2] == 3, "SIMD3 requires depth=3")
    
    let rows = shape[1]
    let cols = shape[0]
    var result: [[[Scalar]]] = []
    
    let scalarVec = SIMD3<Scalar>(repeating: Scalar(scalar))
    
    for d in 0..<3 {
      var depthResult: [[Scalar]] = []
      
      for r in 0..<rows {
        var rowResult: [Scalar] = []
        rowResult.reserveCapacity(cols)
        
        var c = 0
        while c + 2 < cols {
          let vec = SIMD3<Scalar>(
            Scalar(value[d][r][c]),
            Scalar(value[d][r][c+1]),
            Scalar(value[d][r][c+2])
          )
          
          let product = vec * scalarVec
          
          rowResult.append(Scalar(product.x))
          rowResult.append(Scalar(product.y))
          rowResult.append(Scalar(product.z))
          
          c += 3
        }
        
        while c < cols {
          rowResult.append(value[d][r][c] * scalar)
          c += 1
        }
        
        depthResult.append(rowResult)
      }
      result.append(depthResult)
    }
    
    return Tensor(result, context: context)
  }
  
  /// Fast scalar division for depth-3 tensors
  internal func divideSIMD3Scalar(_ scalar: Scalar) -> Tensor {
    precondition(shape[2] == 3, "SIMD3 requires depth=3")
    
    let rows = shape[1]
    let cols = shape[0]
    var result: [[[Scalar]]] = []
    
    let scalarVec = SIMD3<Scalar>(repeating: Scalar(scalar))
    
    for d in 0..<3 {
      var depthResult: [[Scalar]] = []
      
      for r in 0..<rows {
        var rowResult: [Scalar] = []
        rowResult.reserveCapacity(cols)
        
        var c = 0
        while c + 2 < cols {
          let vec = SIMD3<Scalar>(
            Scalar(value[d][r][c]),
            Scalar(value[d][r][c+1]),
            Scalar(value[d][r][c+2])
          )
          
          let quotient = vec / scalarVec
          
          rowResult.append(Scalar(quotient.x))
          rowResult.append(Scalar(quotient.y))
          rowResult.append(Scalar(quotient.z))
          
          c += 3
        }
        
        while c < cols {
          rowResult.append(value[d][r][c] / scalar)
          c += 1
        }
        
        depthResult.append(rowResult)
      }
      result.append(depthResult)
    }
    
    return Tensor(result, context: context)
  }
}

// MARK: - SIMD Operations for Depth=4 (RGBA, Small Feature Maps)

extension Tensor {
  
  /// Fast element-wise addition for depth-4 tensors using SIMD4
  internal func addSIMD4(_ rhs: Tensor) -> Tensor {
    precondition(shape[2] == 4 && rhs.shape[2] == 4, "SIMD4 requires depth=4")
    
    let rows = shape[1]
    let cols = shape[0]
    var result: [[[Scalar]]] = []
    
    for d in 0..<4 {
      var depthResult: [[Scalar]] = []
      
      for r in 0..<rows {
        var rowResult: [Scalar] = []
        rowResult.reserveCapacity(cols)
        
        var c = 0
        while c + 3 < cols {
          let lhsVec = SIMD4<Scalar>(
            Scalar(value[d][r][c]),
            Scalar(value[d][r][c+1]),
            Scalar(value[d][r][c+2]),
            Scalar(value[d][r][c+3])
          )
          let rhsVec = SIMD4<Scalar>(
            Scalar(rhs.value[d][r][c]),
            Scalar(rhs.value[d][r][c+1]),
            Scalar(rhs.value[d][r][c+2]),
            Scalar(rhs.value[d][r][c+3])
          )
          
          let sum = lhsVec + rhsVec
          
          rowResult.append(Scalar(sum.x))
          rowResult.append(Scalar(sum.y))
          rowResult.append(Scalar(sum.z))
          rowResult.append(Scalar(sum.w))
          
          c += 4
        }
        
        while c < cols {
          rowResult.append(value[d][r][c] + rhs.value[d][r][c])
          c += 1
        }
        
        depthResult.append(rowResult)
      }
      result.append(depthResult)
    }
    
    return Tensor(result, context: context)
  }
  
  /// Fast element-wise subtraction for depth-4 tensors using SIMD4
  internal func subtractSIMD4(_ rhs: Tensor) -> Tensor {
    precondition(shape[2] == 4 && rhs.shape[2] == 4, "SIMD4 requires depth=4")
    
    let rows = shape[1]
    let cols = shape[0]
    var result: [[[Scalar]]] = []
    
    for d in 0..<4 {
      var depthResult: [[Scalar]] = []
      
      for r in 0..<rows {
        var rowResult: [Scalar] = []
        rowResult.reserveCapacity(cols)
        
        var c = 0
        while c + 3 < cols {
          let lhsVec = SIMD4<Scalar>(
            Scalar(value[d][r][c]),
            Scalar(value[d][r][c+1]),
            Scalar(value[d][r][c+2]),
            Scalar(value[d][r][c+3])
          )
          let rhsVec = SIMD4<Scalar>(
            Scalar(rhs.value[d][r][c]),
            Scalar(rhs.value[d][r][c+1]),
            Scalar(rhs.value[d][r][c+2]),
            Scalar(rhs.value[d][r][c+3])
          )
          
          let difference = lhsVec - rhsVec
          
          rowResult.append(Scalar(difference.x))
          rowResult.append(Scalar(difference.y))
          rowResult.append(Scalar(difference.z))
          rowResult.append(Scalar(difference.w))
          
          c += 4
        }
        
        while c < cols {
          rowResult.append(value[d][r][c] - rhs.value[d][r][c])
          c += 1
        }
        
        depthResult.append(rowResult)
      }
      result.append(depthResult)
    }
    
    return Tensor(result, context: context)
  }
  
  /// Fast element-wise multiplication for depth-4 tensors using SIMD4
  internal func multiplySIMD4(_ rhs: Tensor) -> Tensor {
    precondition(shape[2] == 4 && rhs.shape[2] == 4, "SIMD4 requires depth=4")
    
    let rows = shape[1]
    let cols = shape[0]
    var result: [[[Scalar]]] = []
    
    for d in 0..<4 {
      var depthResult: [[Scalar]] = []
      
      for r in 0..<rows {
        var rowResult: [Scalar] = []
        rowResult.reserveCapacity(cols)
        
        var c = 0
        while c + 3 < cols {
          let lhsVec = SIMD4<Scalar>(
            Scalar(value[d][r][c]),
            Scalar(value[d][r][c+1]),
            Scalar(value[d][r][c+2]),
            Scalar(value[d][r][c+3])
          )
          let rhsVec = SIMD4<Scalar>(
            Scalar(rhs.value[d][r][c]),
            Scalar(rhs.value[d][r][c+1]),
            Scalar(rhs.value[d][r][c+2]),
            Scalar(rhs.value[d][r][c+3])
          )
          
          let product = lhsVec * rhsVec
          
          rowResult.append(Scalar(product.x))
          rowResult.append(Scalar(product.y))
          rowResult.append(Scalar(product.z))
          rowResult.append(Scalar(product.w))
          
          c += 4
        }
        
        while c < cols {
          rowResult.append(value[d][r][c] * rhs.value[d][r][c])
          c += 1
        }
        
        depthResult.append(rowResult)
      }
      result.append(depthResult)
    }
    
    return Tensor(result, context: context)
  }
  
  /// Fast element-wise division for depth-4 tensors using SIMD4
  internal func divideSIMD4(_ rhs: Tensor) -> Tensor {
    precondition(shape[2] == 4 && rhs.shape[2] == 4, "SIMD4 requires depth=4")
    
    let rows = shape[1]
    let cols = shape[0]
    var result: [[[Scalar]]] = []
    
    for d in 0..<4 {
      var depthResult: [[Scalar]] = []
      
      for r in 0..<rows {
        var rowResult: [Scalar] = []
        rowResult.reserveCapacity(cols)
        
        var c = 0
        while c + 3 < cols {
          let lhsVec = SIMD4<Scalar>(
            Scalar(value[d][r][c]),
            Scalar(value[d][r][c+1]),
            Scalar(value[d][r][c+2]),
            Scalar(value[d][r][c+3])
          )
          let rhsVec = SIMD4<Scalar>(
            Scalar(rhs.value[d][r][c]),
            Scalar(rhs.value[d][r][c+1]),
            Scalar(rhs.value[d][r][c+2]),
            Scalar(rhs.value[d][r][c+3])
          )
          
          let quotient = lhsVec / rhsVec
          
          rowResult.append(Scalar(quotient.x))
          rowResult.append(Scalar(quotient.y))
          rowResult.append(Scalar(quotient.z))
          rowResult.append(Scalar(quotient.w))
          
          c += 4
        }
        
        while c < cols {
          rowResult.append(value[d][r][c] / rhs.value[d][r][c])
          c += 1
        }
        
        depthResult.append(rowResult)
      }
      result.append(depthResult)
    }
    
    return Tensor(result, context: context)
  }
  
  /// Fast scalar addition for depth-4 tensors
  internal func addSIMD4Scalar(_ scalar: Scalar) -> Tensor {
    precondition(shape[2] == 4, "SIMD4 requires depth=4")
    
    let rows = shape[1]
    let cols = shape[0]
    var result: [[[Scalar]]] = []
    
    let scalarVec = SIMD4<Scalar>(repeating: Scalar(scalar))
    
    for d in 0..<4 {
      var depthResult: [[Scalar]] = []
      
      for r in 0..<rows {
        var rowResult: [Scalar] = []
        rowResult.reserveCapacity(cols)
        
        var c = 0
        while c + 3 < cols {
          let vec = SIMD4<Scalar>(
            Scalar(value[d][r][c]),
            Scalar(value[d][r][c+1]),
            Scalar(value[d][r][c+2]),
            Scalar(value[d][r][c+3])
          )
          
          let sum = vec + scalarVec
          
          rowResult.append(Scalar(sum.x))
          rowResult.append(Scalar(sum.y))
          rowResult.append(Scalar(sum.z))
          rowResult.append(Scalar(sum.w))
          
          c += 4
        }
        
        while c < cols {
          rowResult.append(value[d][r][c] + scalar)
          c += 1
        }
        
        depthResult.append(rowResult)
      }
      result.append(depthResult)
    }
    
    return Tensor(result, context: context)
  }
  
  /// Fast scalar subtraction for depth-4 tensors
  internal func subtractSIMD4Scalar(_ scalar: Scalar) -> Tensor {
    precondition(shape[2] == 4, "SIMD4 requires depth=4")
    
    let rows = shape[1]
    let cols = shape[0]
    var result: [[[Scalar]]] = []
    
    let scalarVec = SIMD4<Scalar>(repeating: Scalar(scalar))
    
    for d in 0..<4 {
      var depthResult: [[Scalar]] = []
      
      for r in 0..<rows {
        var rowResult: [Scalar] = []
        rowResult.reserveCapacity(cols)
        
        var c = 0
        while c + 3 < cols {
          let vec = SIMD4<Scalar>(
            Scalar(value[d][r][c]),
            Scalar(value[d][r][c+1]),
            Scalar(value[d][r][c+2]),
            Scalar(value[d][r][c+3])
          )
          
          let difference = vec - scalarVec
          
          rowResult.append(Scalar(difference.x))
          rowResult.append(Scalar(difference.y))
          rowResult.append(Scalar(difference.z))
          rowResult.append(Scalar(difference.w))
          
          c += 4
        }
        
        while c < cols {
          rowResult.append(value[d][r][c] - scalar)
          c += 1
        }
        
        depthResult.append(rowResult)
      }
      result.append(depthResult)
    }
    
    return Tensor(result, context: context)
  }
  
  /// Fast scalar multiplication for depth-4 tensors
  internal func multiplySIMD4Scalar(_ scalar: Scalar) -> Tensor {
    precondition(shape[2] == 4, "SIMD4 requires depth=4")
    
    let rows = shape[1]
    let cols = shape[0]
    var result: [[[Scalar]]] = []
    
    let scalarVec = SIMD4<Scalar>(repeating: Scalar(scalar))
    
    for d in 0..<4 {
      var depthResult: [[Scalar]] = []
      
      for r in 0..<rows {
        var rowResult: [Scalar] = []
        rowResult.reserveCapacity(cols)
        
        var c = 0
        while c + 3 < cols {
          let vec = SIMD4<Scalar>(
            Scalar(value[d][r][c]),
            Scalar(value[d][r][c+1]),
            Scalar(value[d][r][c+2]),
            Scalar(value[d][r][c+3])
          )
          
          let product = vec * scalarVec
          
          rowResult.append(Scalar(product.x))
          rowResult.append(Scalar(product.y))
          rowResult.append(Scalar(product.z))
          rowResult.append(Scalar(product.w))
          
          c += 4
        }
        
        while c < cols {
          rowResult.append(value[d][r][c] * scalar)
          c += 1
        }
        
        depthResult.append(rowResult)
      }
      result.append(depthResult)
    }
    
    return Tensor(result, context: context)
  }
  
  /// Fast scalar division for depth-4 tensors
  internal func divideSIMD4Scalar(_ scalar: Scalar) -> Tensor {
    precondition(shape[2] == 4, "SIMD4 requires depth=4")
    
    let rows = shape[1]
    let cols = shape[0]
    var result: [[[Scalar]]] = []
    
    let scalarVec = SIMD4<Scalar>(repeating: Scalar(scalar))
    
    for d in 0..<4 {
      var depthResult: [[Scalar]] = []
      
      for r in 0..<rows {
        var rowResult: [Scalar] = []
        rowResult.reserveCapacity(cols)
        
        var c = 0
        while c + 3 < cols {
          let vec = SIMD4<Scalar>(
            Scalar(value[d][r][c]),
            Scalar(value[d][r][c+1]),
            Scalar(value[d][r][c+2]),
            Scalar(value[d][r][c+3])
          )
          
          let quotient = vec / scalarVec
          
          rowResult.append(Scalar(quotient.x))
          rowResult.append(Scalar(quotient.y))
          rowResult.append(Scalar(quotient.z))
          rowResult.append(Scalar(quotient.w))
          
          c += 4
        }
        
        while c < cols {
          rowResult.append(value[d][r][c] / scalar)
          c += 1
        }
        
        depthResult.append(rowResult)
      }
      result.append(depthResult)
    }
    
    return Tensor(result, context: context)
  }
}

// MARK: - SIMD Activation Functions

extension Tensor {
  
  /// Fast ReLU using SIMD4 for depth≤4 tensors
  internal func reluSIMD() -> Tensor {
    let depth = shape[2]
    precondition(depth <= 4, "SIMD ReLU optimized for depth≤4")
    
    let rows = shape[1]
    let cols = shape[0]
    var result: [[[Scalar]]] = []
    
    let zero = SIMD4<Scalar>(repeating: 0)
    
    for d in 0..<depth {
      var depthResult: [[Scalar]] = []
      
      for r in 0..<rows {
        var rowResult: [Scalar] = []
        rowResult.reserveCapacity(cols)
        
        var c = 0
        while c + 3 < cols {
          let vec = SIMD4<Scalar>(
            Scalar(value[d][r][c]),
            Scalar(value[d][r][c+1]),
            Scalar(value[d][r][c+2]),
            Scalar(value[d][r][c+3])
          )
          
          // max(0, x) - SIMD instruction
          let activated = simd_max(vec, zero)
          
          rowResult.append(Scalar(activated.x))
          rowResult.append(Scalar(activated.y))
          rowResult.append(Scalar(activated.z))
          rowResult.append(Scalar(activated.w))
          
          c += 4
        }
        
        while c < cols {
          rowResult.append(max(0, value[d][r][c]))
          c += 1
        }
        
        depthResult.append(rowResult)
      }
      result.append(depthResult)
    }
    
    return Tensor(result, context: context)
  }
  
  /// Fast Tanh approximation using SIMD
  internal func tanhSIMD() -> Tensor {
    let depth = shape[2]
    precondition(depth <= 4, "SIMD Tanh optimized for depth≤4")
    
    let rows = shape[1]
    let cols = shape[0]
    var result: [[[Scalar]]] = []
    
    for d in 0..<depth {
      var depthResult: [[Scalar]] = []
      
      for r in 0..<rows {
        var rowResult: [Scalar] = []
        rowResult.reserveCapacity(cols)
        
        var c = 0
        while c + 3 < cols {
          var vec = SIMD4<Scalar>(
            Scalar(value[d][r][c]),
            Scalar(value[d][r][c+1]),
            Scalar(value[d][r][c+2]),
            Scalar(value[d][r][c+3])
          )
          
          // (e^x - e^-x) / (e^x + e^-x)
          let num = _simd_exp_f4(vec) - _simd_exp_f4(-vec)
          let denom = _simd_exp_f4(vec) + _simd_exp_f4(-vec)
          vec = vec / denom
          
          rowResult.append(Scalar(vec.x))
          rowResult.append(Scalar(vec.y))
          rowResult.append(Scalar(vec.z))
          rowResult.append(Scalar(vec.w))
          
          c += 4
        }
        
        while c < cols {
          let x = Scalar(value[d][r][c])
          let result = (Scalar.exp(x) - Scalar.exp(-x)) / ((Scalar.exp(x) + Scalar.exp(-x)))
          rowResult.append(result)
          c += 1
        }
        
        depthResult.append(rowResult)
      }
      result.append(depthResult)
    }
    
    return Tensor(result, context: context)
  }
  
  /// Fast Sigmoid using SIMD4 for depth≤4 tensors
  internal func sigmoidSIMD() -> Tensor {
    let depth = shape[2]
    precondition(depth <= 4, "SIMD Sigmoid optimized for depth≤4")
    
    let rows = shape[1]
    let cols = shape[0]
    var result: [[[Scalar]]] = []
    
    let one = SIMD4<Scalar>(repeating: 1)
    
    for d in 0..<depth {
      var depthResult: [[Scalar]] = []
      
      for r in 0..<rows {
        var rowResult: [Scalar] = []
        rowResult.reserveCapacity(cols)
        
        var c = 0
        while c + 3 < cols {
          let vec = SIMD4<Scalar>(
            Scalar(value[d][r][c]),
            Scalar(value[d][r][c+1]),
            Scalar(value[d][r][c+2]),
            Scalar(value[d][r][c+3])
          )
          
          // sigmoid(x) = 1 / (1 + exp(-x))
          let negVec = -vec
          let expVec = SIMD4<Scalar>(
            Scalar.exp(negVec.x),
            Scalar.exp(negVec.y),
            Scalar.exp(negVec.z),
            Scalar.exp(negVec.w)
          )
          let activated = one / (one + expVec)
          
          rowResult.append(Scalar(activated.x))
          rowResult.append(Scalar(activated.y))
          rowResult.append(Scalar(activated.z))
          rowResult.append(Scalar(activated.w))
          
          c += 4
        }
        
        while c < cols {
          let x = Scalar(value[d][r][c])
          let sig = Scalar(1.0) / (Scalar(1.0) + Scalar.exp(-x))
          rowResult.append(Scalar(sig))
          c += 1
        }
        
        depthResult.append(rowResult)
      }
      result.append(depthResult)
    }
    
    return Tensor(result, context: context)
  }
  
  /// Fast LeakyReLU using SIMD4 for depth≤4 tensors
  internal func leakyReluSIMD(limit: Scalar) -> Tensor {
    // #if arch(arm64) && QUANTIZED_F16
    // if Scalar.self == Scalar16.self {
    //   return leakyReluSIMD16(limit: limit)
    // }
    // #endif
    
    let depth = shape[2]
    precondition(depth <= 4, "SIMD LeakyReLU optimized for depth≤4")
    
    let rows = shape[1]
    let cols = shape[0]
    var result: [[[Scalar]]] = []
    
    let limitVec = SIMD4<Scalar>(repeating: Scalar(limit))
    
    for d in 0..<depth {
      var depthResult: [[Scalar]] = []
      
      for r in 0..<rows {
        var rowResult: [Scalar] = []
        rowResult.reserveCapacity(cols)
        
        var c = 0
        while c + 3 < cols {
          let vec = SIMD4<Scalar>(
            Scalar(value[d][r][c]),
            Scalar(value[d][r][c+1]),
            Scalar(value[d][r][c+2]),
            Scalar(value[d][r][c+3])
          )
          
          // LeakyReLU: x >= 0 ? x : limit * x
          let limitTimesVec = limitVec * vec
          let activated = SIMD4<Scalar>(
            vec.x >= 0 ? vec.x : limitTimesVec.x,
            vec.y >= 0 ? vec.y : limitTimesVec.y,
            vec.z >= 0 ? vec.z : limitTimesVec.z,
            vec.w >= 0 ? vec.w : limitTimesVec.w
          )
          
          rowResult.append(Scalar(activated.x))
          rowResult.append(Scalar(activated.y))
          rowResult.append(Scalar(activated.z))
          rowResult.append(Scalar(activated.w))
          
          c += 4
        }
        
        while c < cols {
          let x = value[d][r][c]
          rowResult.append(x >= 0 ? x : limit * x)
          c += 1
        }
        
        depthResult.append(rowResult)
      }
      result.append(depthResult)
    }
    
    return Tensor(result, context: context)
  }
  
  /// Fast Swish using SIMD4 for depth≤4 tensors
  internal func swishSIMD() -> Tensor {
    let depth = shape[2]
    precondition(depth <= 4, "SIMD Swish optimized for depth≤4")
    
    let rows = shape[1]
    let cols = shape[0]
    var result: [[[Scalar]]] = []
    
    let one = SIMD4<Scalar>(repeating: 1)
    
    for d in 0..<depth {
      var depthResult: [[Scalar]] = []
      
      for r in 0..<rows {
        var rowResult: [Scalar] = []
        rowResult.reserveCapacity(cols)
        
        var c = 0
        while c + 3 < cols {
          let vec = SIMD4<Scalar>(
            Scalar(value[d][r][c]),
            Scalar(value[d][r][c+1]),
            Scalar(value[d][r][c+2]),
            Scalar(value[d][r][c+3])
          )
          
          // Swish: x * sigmoid(x) = x / (1 + exp(-x))
          let negVec = -vec
          let expVec = SIMD4<Scalar>(
            Scalar.exp(negVec.x),
            Scalar.exp(negVec.y),
            Scalar.exp(negVec.z),
            Scalar.exp(negVec.w),
          )
          let sigmoidVec = one / (one + expVec)
          let activated = vec * sigmoidVec
          
          rowResult.append(Scalar(activated.x))
          rowResult.append(Scalar(activated.y))
          rowResult.append(Scalar(activated.z))
          rowResult.append(Scalar(activated.w))
          
          c += 4
        }
        
        while c < cols {
          let x = Scalar(value[d][r][c])
          let sig = Scalar(1.0) / (Scalar(1.0) + Scalar.exp(-x))
          rowResult.append(Scalar(x * sig))
          c += 1
        }
        
        depthResult.append(rowResult)
      }
      result.append(depthResult)
    }
    
    return Tensor(result, context: context)
  }
  
  /// Fast SELU using SIMD4 for depth≤4 tensors
  internal func seLuSIMD() -> Tensor {
    let depth = shape[2]
    precondition(depth <= 4, "SIMD SELU optimized for depth≤4")
    
    let rows = shape[1]
    let cols = shape[0]
    var result: [[[Scalar]]] = []
    
    let lambda: Scalar = 1.0507
    let alpha: Scalar = 1.6733
    let lambdaVec = SIMD4<Scalar>(repeating: lambda)
    let alphaLambdaVec = SIMD4<Scalar>(repeating: lambda * alpha)
    let one = SIMD4<Scalar>(repeating: 1)
    
    for d in 0..<depth {
      var depthResult: [[Scalar]] = []
      
      for r in 0..<rows {
        var rowResult: [Scalar] = []
        rowResult.reserveCapacity(cols)
        
        var c = 0
        while c + 3 < cols {
          let vec = SIMD4<Scalar>(
            Scalar(value[d][r][c]),
            Scalar(value[d][r][c+1]),
            Scalar(value[d][r][c+2]),
            Scalar(value[d][r][c+3])
          )
          
          // SELU: x > 0 ? lambda * x : lambda * alpha * (exp(x) - 1)
          let expVec = SIMD4<Scalar>(
            Scalar.exp(vec.x),
            Scalar.exp(vec.y),
            Scalar.exp(vec.z),
            Scalar.exp(vec.w)
          )
          let expMinusOne = expVec - one
          let negativeBranch = alphaLambdaVec * expMinusOne
          let positiveBranch = lambdaVec * vec
          let activated = SIMD4<Scalar>(
            vec.x > 0 ? positiveBranch.x : negativeBranch.x,
            vec.y > 0 ? positiveBranch.y : negativeBranch.y,
            vec.z > 0 ? positiveBranch.z : negativeBranch.z,
            vec.w > 0 ? positiveBranch.w : negativeBranch.w
          )
          
          rowResult.append(Scalar(activated.x))
          rowResult.append(Scalar(activated.y))
          rowResult.append(Scalar(activated.z))
          rowResult.append(Scalar(activated.w))
          
          c += 4
        }
        
        while c < cols {
          let x = Scalar(value[d][r][c])
          let activated: Scalar
          if x > 0 {
            activated = lambda * x
          } else {
            activated = lambda * alpha * (Scalar.exp(x) - 1)
          }
          rowResult.append(Scalar(activated))
          c += 1
        }
        
        depthResult.append(rowResult)
      }
      result.append(depthResult)
    }
    
    return Tensor(result, context: context)
  }
  
  /// Fast GELU using SIMD4 for depth≤4 tensors
  internal func geLuSIMD() -> Tensor {
    let depth = shape[2]
    precondition(depth <= 4, "SIMD GELU optimized for depth≤4")
    
    let rows = shape[1]
    let cols = shape[0]
    var result: [[[Scalar]]] = []
    
    let sqrt2: Scalar = Scalar(Foundation.sqrt(2.0))
    let half = SIMD4<Scalar>(repeating: 0.5)
    let one = SIMD4<Scalar>(repeating: 1)
    let sqrt2Vec = SIMD4<Scalar>(repeating: sqrt2)
    
    for d in 0..<depth {
      var depthResult: [[Scalar]] = []
      
      for r in 0..<rows {
        var rowResult: [Scalar] = []
        rowResult.reserveCapacity(cols)
        
        var c = 0
        while c + 3 < cols {
          let vec = SIMD4<Scalar>(
            Scalar(value[d][r][c]),
            Scalar(value[d][r][c+1]),
            Scalar(value[d][r][c+2]),
            Scalar(value[d][r][c+3])
          )
          
          // GELU: x * (1 + erf(x / sqrt(2))) / 2
          let normalized = vec / sqrt2Vec
          let erfVec = SIMD4<Scalar>(
            Scalar.erf(normalized.x),
            Scalar.erf(normalized.y),
            Scalar.erf(normalized.z),
            Scalar.erf(normalized.w)
          )
          let activated = vec * (one + erfVec) * half
          
          rowResult.append(Scalar(activated.x))
          rowResult.append(Scalar(activated.y))
          rowResult.append(Scalar(activated.z))
          rowResult.append(Scalar(activated.w))
          
          c += 4
        }
        
        while c < cols {
          let x = Scalar(value[d][r][c])
          let activated = x * (Scalar(1) + Scalar.erf(x / sqrt2)) * Scalar(0.5)
          rowResult.append(Scalar(activated))
          c += 1
        }
        
        depthResult.append(rowResult)
      }
      result.append(depthResult)
    }
    
    return Tensor(result, context: context)
  }
}

// MARK: - Generic SIMD Operations (Auto-Vectorization)

extension Tensor {
  
  /// Performs element-wise operation using optimal SIMD width
  internal func applyElementWiseSIMD(
    _ rhs: Tensor,
    operation: (SIMD4<Scalar>, SIMD4<Scalar>) -> SIMD4<Scalar>
  ) -> Tensor {
    let rows = shape[1]
    let cols = shape[0]
    let depth = shape[2]
    var result: [[[Scalar]]] = []
    
    for d in 0..<depth {
      var depthResult: [[Scalar]] = []
      
      for r in 0..<rows {
        var rowResult: [Scalar] = []
        rowResult.reserveCapacity(cols)
        
        // Process in SIMD4 chunks
        var c = 0
        while c + 3 < cols {
          let lhsVec = SIMD4<Scalar>(
            Scalar(value[d][r][c]),
            Scalar(value[d][r][c+1]),
            Scalar(value[d][r][c+2]),
            Scalar(value[d][r][c+3])
          )
          let rhsVec = SIMD4<Scalar>(
            Scalar(rhs.value[d][r][c]),
            Scalar(rhs.value[d][r][c+1]),
            Scalar(rhs.value[d][r][c+2]),
            Scalar(rhs.value[d][r][c+3])
          )
          
          let resultVec = operation(lhsVec, rhsVec)
          
          rowResult.append(Scalar(resultVec.x))
          rowResult.append(Scalar(resultVec.y))
          rowResult.append(Scalar(resultVec.z))
          rowResult.append(Scalar(resultVec.w))
          
          c += 4
        }
        
        // Handle remainder (fallback to scalar)
        while c < cols {
          let lhs = SIMD4<Scalar>(Scalar(value[d][r][c]), 0, 0, 0)
          let rhs = SIMD4<Scalar>(Scalar(rhs.value[d][r][c]), 0, 0, 0)
          let result = operation(lhs, rhs)
          rowResult.append(Scalar(result.x))
          c += 1
        }
        
        depthResult.append(rowResult)
      }
      result.append(depthResult)
    }
    
    return Tensor(result, context: context)
  }
}

#endif // arch(arm64) || arch(x86_64)
