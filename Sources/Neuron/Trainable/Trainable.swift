
import Foundation
import NumSwift
import NumSwiftC

/// The base object that organizes a network.
///
/// Currently `Sequential` is the only conformer to this, so it is highly recommended to use that object.
///
/// Get debug data from the `Trainable` by calling `print(sequential)`, or using `lldb`: `po trainable`, where `sequential` is your `Trainable` object.
///
public protocol Trainable: AnyObject, Exportable, CustomDebugStringConvertible {
  
  /// Generic name of the trainable. Used when printing the network
  var name: String { get set }
  
  /// The layers of the network
  var layers: [Layer] { get }
  
  /// Indicates if the network has been setup correctly and is ready for training.
  var isCompiled: Bool { get }
  
  /// Indicates if this particular network has its weights updated. Mainly used for Batch and Layer normalize. As they have different paths for training and not training.
  var isTraining: Bool { get set }
  
  /// The compute device type used for inference and training.
  var deviceType: DeviceType { get set }
  
  /// The device to execute the ML ops and math ops on. Default: CPU()
  var device: Device { get }
  
  /// The current batch size. Default: 1
  var batchSize: Int { get set }
  
  /// Creates a Trainable object from a `.smodel` file.
  /// - Parameter url: The URL to the `.smodel` file.
  /// - Returns: The network built from the file.
  static func `import`(_ url: URL) -> Self
  
  /// Performs a forward pass on the network
  /// - Parameters:
  ///   - data: The input tensor.
  ///   - context: Batch/thread metadata propagated through layers.
  /// - Returns: The output of the network
  func predict(_ data: Tensor, context: NetworkContext) -> Tensor
  
  /// Performs a forward pass on the network
  /// - Parameters:
  ///   - batch: The input batch.
  ///   - context: Batch/thread metadata propagated through layers.
  /// - Returns: An array of outputs
  func predict(batch: TensorBatch, context: NetworkContext) -> TensorBatch 

  /// Compiles the network, getting it ready to be trained.
  func compile()
  
  /// Exports the weights of the network
  func exportWeights() throws -> [[Tensor]]
 
  /// Attempts to replace the weights in the network
  func importWeights(_ weights: [[Tensor]]) throws
  
  /// Applies a gradient payload to all trainable layers.
  ///
  /// - Parameters:
  ///   - gradients: Gradient payload where each index corresponds to a layer.
  ///   - learningRate: Step size used by each layer's update rule.
  func apply(gradients: Tensor.Gradient, learningRate: Tensor.Scalar)

  /// Exports the network to a `.smodel` file.
  ///
  /// - Parameters:
  ///   - name: Optional filename prefix.
  ///   - overrite: When `false`, appends a timestamp to avoid overwriting.
  ///   - compress: When `true`, emits compact JSON.
  /// - Returns: URL to the exported file, or `nil` on failure.
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
  /// Exports the trainable as a `.smodel` file.
  ///
  /// - Parameters:
  ///   - name: Optional filename prefix.
  ///   - overrite: When `false`, appends a timestamp to avoid overwrite.
  ///   - compress: When `true`, emits compact JSON.
  /// - Returns: URL to the exported model file, or `nil` on write failure.
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
  static let col4Width = 9

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
    
    string.append(previousLine.build())
    
    // do we want to support showing skip connection?
    struct LinkLayer {
      var linkId: String
      var slot: Int  // fixed visual column slot (0-based), assigned at insertion and never changed
    }
    
    var layerString: [String] = []
    
    var linkLayers: [LinkLayer] = []
    var linkingSet: Set<String> = []
    var maxSlot: Int = 0  // high-water mark of simultaneous open slots
    
    let totalParametersTensors: [Tensor] = (try? trainable.exportWeights())?.fullFlatten() ?? []
    let totalParameters = totalParametersTensors.reduce(0) { $0 + Int($1.shape.reduce(1, *)) }
    
    for layer in trainable.layers.reversed() {
      // when we hit a layer that has a linkTo, aka a ArithmeticLayer, we store the layer it's linked to
      var hasLinkTo: Bool = false
      if let mathLayer = layer as? ArithmeticLayer {
        linkingSet.insert(mathLayer.linkTo)
        let newSlot = linkLayers.count  // next available slot
        linkLayers.append(.init(linkId: mathLayer.linkTo, slot: newSlot))
        maxSlot = max(maxSlot, newSlot + 1)
        hasLinkTo = true
      }
      
      let currentLink = linkLayers.first(where: { $0.linkId == layer.linkId })
      let layerIsLinked: Bool = currentLink != nil

      let linkType: LinkType = if layerIsLinked {
        .top
      } else if hasLinkTo {
        .bottom
      } else if linkingSet.isEmpty == false {
        .line
      } else {
        .none
      }

      // branchOffset = 1-based slot of the current link's corner (or total open count for .line)
      // dimensions = total width of the connection zone (max simultaneous slots ever open)
      let offset: Int
      if let link = currentLink {
        offset = link.slot + 1  // fixed slot, never changes
      } else if hasLinkTo {
        offset = linkLayers.last!.slot + 1  // slot of the just-appended link
      } else {
        offset = maxSlot  // .line: zone width stays at max
      }
            
      let activeSlots = Set(linkLayers.map { $0.slot })

      let line = line(layer: layer,
                      previousLine: previousLine,
                      branchOffset: offset,
                      dimensions: maxSlot,
                      linkType: linkType,
                      linkId: layer.linkId,
                      activeSlots: activeSlots)
      
      // this means we've completed the link and we should remove it
      if linkType == .top {
        linkingSet.remove(layer.linkId)
        linkLayers.removeAll(where: { $0.linkId == layer.linkId })
      }
      
      let toAdd = spacer + line.build()
      
      layerString.append(toAdd)
      previousLine = line
    }
    
    string.append(layerString.reversed().joined())
     
    string.append(spacer)
    string.append("\nTotal Parameters: \(totalParameters)\n")
    
    return string
  }
  
  static func line(layer: Layer,
                   previousLine: Line? = nil,
                   branchOffset: Int = 1,
                   dimensions: Int = 0,
                   linkType: LinkType = .none,
                   linkId: String = "",
                   activeSlots: Set<Int> = []) -> Line {
    let parameters = layer.weights.storage.count
    
    let col1 = Column(value: layer.encodingType.rawValue, width: col1Width)
    let col2 = Column(value: "\(layer.outputSize.asArray)", width: col2Width)
    let col3 = Column(value: "\(parameters)", width: col3Width, leftAlign: false)
    
    let columns = [col1, col2, col3]
    
    guard case .none = linkType else {
      // Fall through to render connection columns
      return buildWithLinks(columns: columns,
                            branchOffset: branchOffset,
                            dimensions: dimensions,
                            columnWidth: 3,
                            linkType: linkType,
                            activeSlots: activeSlots)
    }

    return Line(columns: columns)
  }

  private static func buildWithLinks(columns: [Column],
                                     branchOffset: Int,
                                     dimensions: Int,
                                     columnWidth: Int,
                                     linkType: LinkType,
                                     activeSlots: Set<Int>) -> Line {
    var columns = columns
    // Total active connection columns = max of open links and current link's level
    let totalColumns = max(dimensions, branchOffset)
    // The corner symbol lands at the column matching the link's level (1-based → index branchOffset-1)
    let cornerIndex = branchOffset - 1

    // Build the full connection zone as a single fixed-width string so symbols align perfectly.
    // Total zone width = col4Width + (totalColumns - 1) * columnWidth.
    // Each "slot" occupies columnWidth chars, with slot 0 occupying col4Width chars.
    // The right-most character of slot i is at index: col4Width - 1 + i * columnWidth
    let totalWidth = col4Width + (totalColumns - 1) * columnWidth
    var buf = [Character](repeating: " ", count: totalWidth)

    func slotEnd(_ slot: Int) -> Int { col4Width - 1 + slot * columnWidth }

    switch linkType {
    case .top:
      // ┐ at this link's fixed slot.
      buf[slotEnd(cornerIndex)] = "┐"
      // If the corner is not at slot 0, draw dashes from slotEnd(0) to the corner
      // so the leftmost dash aligns with where ← connects on the sink row
      if cornerIndex > 0 {
        for i in slotEnd(0)..<slotEnd(cornerIndex) { buf[i] = "-" }
      }
      // Other still-open links get | at their fixed slots (overwrite dashes where needed)
      for slot in activeSlots where slot != cornerIndex { buf[slotEnd(slot)] = "|" }
    case .bottom:
      if cornerIndex == 0 {
        // Single-slot: ← + ┘ at slot 0
        buf[slotEnd(0) - 1] = "←"
        buf[slotEnd(0)] = "┘"
        // Other open links get | (slots > 0)
        for slot in activeSlots where slot != 0 { buf[slotEnd(slot)] = "|" }
      } else {
        // Multi-slot: ← one before slot 0 (matching single-slot position), dashes from slot 0 to corner, ┘ at corner slot
        buf[slotEnd(0) - 1] = "←"
        for i in slotEnd(0)..<slotEnd(cornerIndex) { buf[i] = "-" }
        buf[slotEnd(cornerIndex)] = "┘"
        // Other open links that are NOT in the bridge range get |
        for slot in activeSlots where slot != cornerIndex && slot != 0 { buf[slotEnd(slot)] = "|" }
      }
    default:
      // | at every currently-open slot
      for slot in activeSlots { buf[slotEnd(slot)] = "|" }
    }

    columns.append(.init(value: String(buf), width: totalWidth, leftAlign: true))

    return Line(columns: columns)
  }
  
  enum LinkType {
    case top, bottom, line, none
  }
}
