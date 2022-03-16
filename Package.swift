// swift-tools-version:5.5
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Neuron",
    platforms: [ .iOS(.v13),
                 .tvOS(.v13),
                 .watchOS(.v5),
                 .macOS(.v10_15)],
    products: [
        // Products define the executables and libraries produced by a package, and make them visible to other packages.
        .library(
            name: "Neuron",
            targets: ["Neuron"]),
    ],
    dependencies: [
      .package(url: "https://github.com/wvabrinskas/NumSwift.git", from: "1.1.0"),
      //.package(url: "https://github.com/wvabrinskas/NumSwift.git", .branch("develop")),
      .package(url: "https://github.com/wvabrinskas/Logger.git", from: "1.0.6")
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages which this package depends on.
        .target(
            name: "Neuron",
            dependencies: [
              "NumSwift",
              "Logger"
            ],
            resources: [ .process("Resources") ]),
        
        .testTarget(
            name: "NeuronTests",
            dependencies: ["Neuron"]),
    ]
)

