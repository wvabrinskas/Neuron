// swift-tools-version: 5.7.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Neuron",
    platforms: [ .iOS(.v13),
                 .tvOS(.v13),
                 .watchOS(.v6),
                 .macOS(.v11)],
    products: [
        // Products define the executables and libraries a package produces, and make them visible to other packages.
        .library(
            name: "Neuron",
            targets: ["Neuron"]),
    ],
    dependencies: [
      //.package(path: "../NumSwift"),
      .package(url: "https://github.com/wvabrinskas/NumSwift.git", from: "2.0.10"),
      .package(url: "https://github.com/wvabrinskas/Logger.git", from: "1.0.6")
      //.package(url: "https://github.com/apple/swift-docc-plugin", branch: "main")
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages this package depends on.
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
