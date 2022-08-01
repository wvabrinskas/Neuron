
import Foundation
import NumSwift
import NumSwiftC

public protocol Trainable: Codable, CustomDebugStringConvertible {
  var name: String { get set }
  var layers: [Layer] { get }
  var isCompiled: Bool { get }
  var trainable: Bool { get set }
  var device: Device { get set }
  
  static func `import`(_ url: URL) -> Self
  func predict(_ data: Tensor) -> Tensor
  func compile()
}

public extension Trainable {
  var debugDescription: String {
    let string = TrainablePrinter.build(self)
    return string
  }
}

private struct TrainablePrinter {
  static let col1Width = 20
  static let col2Width = 15
  static let col3Width = 10
  
  struct Column {
    var value: String
    var width: Int
    
    init(value: String, width: Int, leftAlign: Bool = true) {
      self.value = value.fill(max: width, leftAlign: leftAlign)
      self.width = width
    }
  }
  
  struct Line {
    var columns: [Column]
    
    func build() -> String {
      return columns.map { $0.value }.joined() + "\n"
    }
  }
  
  static func build(_ trainable: Trainable) -> String {
    var string = """
                 Model: "\(trainable.name)" \n\n
                 """
    
    let spacer = "".fill(with: "-", max: col1Width + col2Width + col3Width + 5) + "\n"
    string.append(spacer)
    
    let col1 = Column(value: "Layer", width: col1Width)
    let col2 = Column(value: "Output Shape", width: col2Width)
    let col3 = Column(value: "Param #", width: col3Width, leftAlign: false)
    
    var previousLine: Line = Line(columns: [col1, col2, col3])
    var totalParameters: Int = 0
    
    string.append(previousLine.build())
    
    for layer in trainable.layers {
      let line = line(layer: layer, previousLine: previousLine)
      if let lastLineParam = Int((line.columns.last?.value ?? "").replacingOccurrences(of: " ", with: "")) {
        totalParameters += lastLineParam
      }
      
      string.append(spacer)
      string.append(line.build())
      previousLine = line
    }
    
    string.append(spacer)
    string.append("\nTotal Parameters: \(totalParameters)\n")
    
    return string
  }
  
  static func line(layer: Layer, previousLine: Line? = nil) -> Line {
    var parameters = layer.weights.value.flatten().count
    
    // TODO: maybe find a better way to do this so we can just reference a property like `parameters` or something
    if let conv = layer as? ConvolutionalLayer {
      parameters = conv.filters.map { $0.value.flatten().count }.sumSlow
    }
    
    let col1 = Column(value: layer.encodingType.rawValue, width: col1Width)
    let col2 = Column(value: "\(layer.outputSize)", width: col2Width)
    let col3 = Column(value: "\(parameters)", width: col3Width, leftAlign: false)

    return Line(columns: [col1, col2, col3])
  }
}
