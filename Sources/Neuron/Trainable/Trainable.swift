
import Foundation
import NumSwift
import NumSwiftC

/// The base object that organizes a network.
///
/// Currently `Sequential` is the only conformer to this, so it is highly recommended to use that object.
///
/// Get debug data from the `Trainable` by calling `print(sequential)`, or using `lldb`: `po trainable`, where `sequential` is your `Trainable` object.
///
/// example:
///
/// Model: "Sequential"
///
/// --------------------------------------------------
/// Layer               Output Shape      Param #
/// --------------------------------------------------
/// dense               TensorSize(rows: 1, columns: 32, depth: 1)      1024
/// --------------------------------------------------
/// leakyRelu           TensorSize(rows: 1, columns: 32, depth: 1)         0
/// --------------------------------------------------
/// dense               TensorSize(rows: 1, columns: 7, depth: 1)       224
/// --------------------------------------------------
/// tanh                TensorSize(rows: 1, columns: 7, depth: 1)         0
/// --------------------------------------------------
///
/// Total Parameters: 1248
///
///
public protocol Trainable: Codable, CustomDebugStringConvertible {
  
  /// Generic name of the trainable. Used when printing the network
  var name: String { get set }
  
  /// The layers of the network
  var layers: [Layer] { get }
  
  /// Indicates if the network has been setup correctly and is ready for training.
  var isCompiled: Bool { get }
  
  /// Indicates if this particular network has its weights updated. Mainly used for Batch and Layer normalize. As they have different paths for training and not training.
  var trainable: Bool { get set }
  
  /// The device to execute the ML ops and math ops on. Default: CPU()
  var device: Device { get set }
  
  /// Creates a Trainable object from a `.smodel` file.
  /// - Parameter url: The URL to the `.smodel` file.
  /// - Returns: The network built from the file.
  static func `import`(_ url: URL) -> Self
  
  /// Performs a forward pass on the network
  /// - Parameter data: The inputs
  /// - Returns: The output of the network
  func predict(_ data: Tensor) -> Tensor
  
  /// Compiles the network, getting it ready to be trained.
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
