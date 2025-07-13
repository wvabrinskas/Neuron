// swift-tools-version: 5.10.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Neuron",
    platforms: [ .iOS(.v14),
                 .tvOS(.v14),
                 .watchOS(.v7),
                 .macOS(.v14)],
    products: [
        .library(
            name: "Neuron",
            targets: ["Neuron"])
    ],
    dependencies: [
      .package(path: "../NumSwift"),
      //.package(url: "https://github.com/wvabrinskas/NumSwift.git", from: "2.0.13"),
      .package(url: "https://github.com/wvabrinskas/Logger.git", from: "1.0.6"),
      .package(url: "https://github.com/apple/swift-numerics", from: "1.0.0"),
      //.package(url: "https://github.com/apple/swift-docc-plugin", branch: "main")
    ],
    targets: [
        .target(
            name: "Neuron",
            dependencies: [
              "NumSwift",
              "Logger",
              .product(name: "Numerics", package: "swift-numerics"),
            ],
            resources: [ .process("Resources") ]),
        .testTarget(
            name: "NeuronTests",
            dependencies: ["Neuron"],
            resources: [ .process("Resources") ]),

    ]
)
