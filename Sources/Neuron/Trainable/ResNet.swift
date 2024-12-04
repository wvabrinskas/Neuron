import Foundation

public struct ResidualBlockConfiguration {
  public let filters: Int
  public let strides: (rows: Int, columns: Int)
  public let filterSize: (rows: Int, columns: Int)
  
  public init(filters: Int,
              strides: (rows: Int, columns: Int),
              filterSize: (rows: Int, columns: Int) = (3,3)) {
    self.filters = filters
    self.strides = strides
    self.filterSize = filterSize
  }
}

public struct ResNetConfiguration {
  public let classes: Int
  public let resBlocks: [ResidualBlockConfiguration]
  public let convolutionConfiguration: ResConvolutionalBlockConfiguration
  
  public init(classes: Int,
              resBlocks: [ResidualBlockConfiguration],
              convolutionConfiguration: ResConvolutionalBlockConfiguration) {
    self.classes = classes
    self.resBlocks = resBlocks
    self.convolutionConfiguration = convolutionConfiguration
  }
}

public final class ResNet: BaseTrainable {
  let configuration: ResNetConfiguration
  var residualBlocks: [ResidualBlock] = []
  
  public init(configuration: ResNetConfiguration) {
    self.configuration = configuration
    super.init()
  }
  
  required convenience public init(from decoder: Decoder) throws {
    fatalError("init(from:) has not been implemented")
  }
  
  public override func predict(_ data: Tensor, context: NetworkContext) -> Tensor {
    .init()
  }
  
  // MARK: Private
  
  public override func compile() {
    var newLayers: [Layer] = []
    
    // initial convolutional layer
    let convConfig = configuration.convolutionConfiguration
    
    let conv1Block = Conv2d(filterCount: convConfig.filters,
                            inputSize: convConfig.inputSize,
                            strides: convConfig.strides,
                            padding: .same,
                            filterSize: convConfig.filterSize,
                            initializer: .heNormal,
                            biasEnabled: true)
    
    let batchNorm1 = BatchNormalize()
    let relu = ReLu()
    let maxPooling1 = MaxPool()
    
    newLayers.append(contentsOf: [conv1Block, batchNorm1, relu, maxPooling1])
    
    // residual layers
    for resBlockConfig in configuration.resBlocks {
      let block = ResidualBlock(config: resBlockConfig,
                                inputSize: maxPooling1.outputSize)
      
      children.append(block.shortcut)
      
      residualBlocks.append(block)
    }
    
    // appending all layers so we can do weight managemenet automatically using the optimizer
    let allResidualLayers = residualBlocks.flatMap { $0.allLayers }
    newLayers.append(contentsOf: allResidualLayers)
    
    // Average pooling and final fully connected layer
    
    
    // init Sequential
    layers = newLayers
    
    isCompiled = true
  }
  
}
