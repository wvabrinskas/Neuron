import QuartzCore

public extension CGFloat {
  func map(from: ClosedRange<CGFloat>, to: ClosedRange<CGFloat>) -> CGFloat {
    let result = ((self - from.lowerBound) / (from.upperBound - from.lowerBound)) * (to.upperBound - to.lowerBound) + to.lowerBound
    return result
  }
}

public extension Double {
  func map(from: ClosedRange<CGFloat>, to: ClosedRange<CGFloat>) -> Double {
    return Double(CGFloat(self).map(from: from, to: to))
  }

}

public extension Float {
  func map(from: ClosedRange<CGFloat>, to: ClosedRange<CGFloat>) -> Float {
    return Float(CGFloat(self).map(from: from, to: to))
  }

  func percent(_ accuracy: Float = 100) -> Self {
    let percent = self * accuracy
    let rounded = roundf(percent * 10) / 10.0
    return max(0, min(accuracy, rounded))
  }
  
}


