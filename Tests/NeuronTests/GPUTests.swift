import XCTest
import NumSwift
@testable import Neuron


class GPUTests: XCTestCase {
  
  func testGPUActivationDerivatives() {
    let inputSize = TensorSize(rows: 1, columns: 64, depth: 1)

    let inputTensor = Tensor([-0.06927669, 0.060528696, 0.13386068, -1.4068661, -0.2558472, -0.86038524, -0.01483807, -0.7885369, -0.5649531, 0.5317644, -0.034257904, 0.41748428, -1.7654481, -0.59165764, 0.44765687, -0.3641865, -0.80734724, 0.08079338, -1.196672, -1.8896624, 0.8678287, -0.089690804, 0.5268547, 1.0992858, -0.35282227, 0.6808125, -0.36301154, -0.9427554, -0.05026886, -0.64333963, 0.11184603, -0.7053062, 0.78735495, -0.7752949, 0.34997648, 0.15354842, -1.8700223, -0.091756314, -0.7133802, -0.59410787, -0.027771592, -1.2697736, -0.4616077, -1.6526673, 0.0009351368, -0.4505347, -1.4047865, -0.28908694, 0.96802247, 0.94833493, -0.035656035, 0.602946, 0.19938767, -0.76528037, -0.3883427, -0.9611606, -1.680434, 0.0964531, -0.5187919, 1.52724, 1.1071575, 1.5249776, -0.5573894, -0.26070985])
    
    let activation = LeakyReLu(limit: 0.2)
    activation.inputSize = inputSize
    activation.device = GPU()
    
    let out = activation.forward(tensor: inputTensor)
    out.setGraph(inputTensor)
    
    let gradientTensor = Tensor([0.071826115,-0.056497753,-0.014605803,-0.35169736,0.047841147,0.19482295,-0.074558735,-0.054916285,-0.11066729,0.05687909,-0.33761844,-0.26777357,-0.22225066,-0.053271674,0.31470385,0.047749296,-0.032442596,-0.19093607,0.080520645,0.10718667,0.01760576,0.25035098,0.09334383,-0.11948272,0.053249188,-0.20150048,-0.024755258,-0.113265894,-0.29031768,-0.34720176,0.009279659,-0.075579755,0.20814607,-0.03356082,0.043104313,0.22337791,-0.20086055,0.058413833,0.23470247,-0.0558414,-0.0686687,-0.050625872,0.12491107,0.075727776,0.06367954,-0.20119214,0.086780235,0.19949502,-0.14165258,-0.0818369,-0.28240415,0.07306895,0.028770866,-0.11303962,0.30811813,-0.24076009,0.20105837,-0.06701031,0.08188536,0.20433265,0.18240416,0.1647326,-0.092384666,-0.25218746])
    
    /*
     [0.014365223, -0.0112995505, -0.0029211608, -0.07033947, 0.009568229, 0.038964592, -0.014911748, -0.010983258, -0.022133458, 0.0113758175, -0.06752369, -0.053554714, -0.04445013, -0.010654335, 0.06294077, 0.009549859, -0.0064885193, -0.038187217, 0.01610413, 0.021437334, 0.0035211518, 0.050070196, 0.018668767, -0.023896543, 0.010649838, -0.040300097, -0.004951052, -0.02265318, -0.058063537, -0.06944036, 0.0018559318, -0.015115951, 0.041629214, -0.0067121643, 0.008620863, 0.044675585, -0.04017211, 0.0116827665, 0.046940494, -0.011168281, -0.01373374, -0.010125174, 0.024982214, 0.015145555, 0.012735908, -0.04023843, 0.017356047, 0.039899003, -0.028330518, -0.016367381, -0.056480832, 0.0146137895, 0.0057541733, -0.022607924, 0.06162363, -0.04815202, 0.040211674, -0.0134020625, 0.016377073, 0.04086653, 0.036480833, 0.032946523, -0.018476933, -0.05043749]*/
    let g = out.gradients(delta: gradientTensor)
    print(g)
  }
  
  func testGPUActivationFunction() {
    let inputSize = TensorSize(rows: 1, columns: 10, depth: 1)
    
    let inputTensor = Tensor([-1, -1, -1, -1, 1, -1, -1, -1, -1, -1])
    
    let activation = ReLu()
    activation.inputSize = inputSize
    activation.device = GPU()
    
    let out = activation.forward(tensor: inputTensor)
    
    XCTAssertEqual(out.shape, [inputSize.columns, inputSize.rows, inputSize.depth])
    XCTAssertTrue(Tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).isValueEqual(to: out))
  }
  
  func testConv2dGPU_Texture() {
    let inputSize = TensorSize(rows: 16, columns: 16, depth: 8)
    
    let inputTensor = Tensor([[[Float]]].init(repeating: [[Float]].init(repeating: [Float].init(repeating: 1,
                                                                                                count: inputSize.columns),
                                                                        count: inputSize.rows),
                                              count: inputSize.depth))
    
    let filterSize = (3, 3, inputSize.depth)
    let filter = Tensor([[[Float]]].init(repeating: [[Float]].init(repeating: [0,1,0],
                                                                   count: filterSize.0),
                                         count: filterSize.2))
    
    let filters = [Tensor].init(repeating: filter, count: 64)
    
    let manager = GPUManager()
    let padding: NumSwift.ConvPadding = .same
    let filterSizeMap = (filterSize.0, filterSize.1)
    let strides = (1,1)
    
    let out = manager.conv2d(inputTensor,
                             filters: filters,
                             padding: padding,
                             filterSize: filterSizeMap,
                             strides: strides,
                             inputSize: inputSize)
    
    let cpuOut = Conv2d(filterCount: filters.count,
                        inputSize: inputSize,
                        strides: strides,
                        padding: padding,
                        filterSize: filterSizeMap,
                        initializer: .heNormal,
                        biasEnabled: false)
    
    cpuOut.filters = filters
    
    let cpuOutVal = cpuOut.forward(tensor: inputTensor)
    
    XCTAssert(out.isValueEqual(to: cpuOutVal))
    
  }
}
