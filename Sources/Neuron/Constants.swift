//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/13/24.
//

import Foundation

/// A namespace for global configuration constants used throughout the framework.
public struct Constants {
/// The default weight initializer type applied when initializing neural network layers.
  ///
  /// Defaults to `.heNormal`, which is suitable for layers using ReLU activations.
  public static var defaultInitializer: InitializerType = .heNormal
  
/// The maximum number of worker threads to use for parallel operations.
  ///
  /// Determined at runtime by querying the number of performance CPU cores and rounding
  /// down to the nearest power of two to allow even work distribution across threads.
  /// Falls back to `4` if the core count cannot be determined.
  public static var maxWorkers: Int = {
    if let perfCores = getSysctlIntValue("hw.perflevel0.physicalcpu") {
      // find closest power of 2 as batch sizes are usually broken up in powers of 2. Eg. 16, 32, 64
      // this allows for a better split of the work between threads as it's evenly divisible.
      return 1 << (Int(log2(Double(perfCores))))
    } else {
      return 4
    }
  }()
  
  static func getSysctlIntValue(_ name: String) -> Int? {
    var size = 0
    sysctlbyname(name, nil, &size, nil, 0)
    var result = 0
    let resultPointer = UnsafeMutableRawPointer.allocate(byteCount: size, alignment: MemoryLayout<Int>.alignment)
    defer { resultPointer.deallocate() }
    if sysctlbyname(name, resultPointer, &size, nil, 0) == 0 {
      result = resultPointer.load(as: Int.self)
      return result
    }
    return nil
  }
}
