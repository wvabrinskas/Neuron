//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/13/24.
//

import Foundation

public struct Constants {
  public static var defaultInitializer: InitializerType = .heNormal
  
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
