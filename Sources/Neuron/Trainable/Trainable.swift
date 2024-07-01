
import Foundation
import NumSwift
import NumSwiftC

/// The base object that organizes a network.
///
/// Currently `Sequential` is the only conformer to this, so it is highly recommended to use that object.
///
/// Get debug data from the `Trainable` by calling `print(sequential)`, or using `lldb`: `po trainable`, where `sequential` is your `Trainable` object.
///
public protocol Trainable: AnyObject, Codable, CustomDebugStringConvertible {
  
  /// The id for the current thread
  var threadId: Int { get set }
  
  /// Generic name of the trainable. Used when printing the network
  var name: String { get set }
  
  /// The layers of the network
  var layers: [Layer] { get }
  
  /// Indicates if the network has been setup correctly and is ready for training.
  var isCompiled: Bool { get }
  
  /// Indicates if this particular network has its weights updated. Mainly used for Batch and Layer normalize. As they have different paths for training and not training.
  var isTraining: Bool { get set }
  
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
  
  /// Exports the weights of the network
  func exportWeights() throws -> [[Tensor]]
 
  /// Attempts to replace the weights in the network
  func importWeights(_ weights: [[Tensor]]) throws
  
  /// Exports network
  @discardableResult
  func export(name: String?, overrite: Bool, compress: Bool) -> URL?
}

public extension Trainable {
  var debugDescription: String {
    guard isCompiled else {
      return "Trainable isn't compiled yet. Please compile first."
    }
    let string = TrainablePrinter.build(self)
    return string
  }
  
  @discardableResult
  func export(name: String? = nil, overrite: Bool = false, compress: Bool = true) -> URL? {
    let additional = overrite == false ? "-\(Date().timeIntervalSince1970)" : ""
    
    let filename = (name ?? "sequential") + additional
    
    let dUrl = ExportHelper.getModel(filename: filename, compress: compress, model: self)
    
    return dUrl
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
    } else if let lstm = layer as? LSTM {
      parameters = lstm.forgetGateWeights.concat(lstm.gateGateWeights).concat(lstm.hiddenOutputWeights).concat(lstm.inputGateWeights).concat(lstm.outputGateWeights).value.flatten().count
    }
    
    let col1 = Column(value: layer.encodingType.rawValue, width: col1Width)
    let col2 = Column(value: "\(layer.outputSize.asArray)", width: col2Width)
    let col3 = Column(value: "\(parameters)", width: col3Width, leftAlign: false)

    return Line(columns: [col1, col2, col3])
  }
}
