// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription
import CompilerPluginSupport

let package = Package(
    name: "Neuron",
    platforms: [ .iOS(.v13),
                 .tvOS(.v13),
                 .watchOS(.v6),
                 .macOS(.v11)],
    products: [
        // Products define the executables and libraries a package produces, and make them visible to other packages.
      .library(
          name: "MacrosDef",
          targets: ["MacrosDef"]
      ),
      
        .library(
            name: "Neuron",
            targets: ["Neuron"]),
    ],
    dependencies: [
      .package(url: "https://github.com/apple/swift-syntax.git", from: "509.0.0-swift-5.9-DEVELOPMENT-SNAPSHOT-2023-04-25-b"),
      .package(url: "https://github.com/wvabrinskas/NumSwift.git", from: "2.0.1"),
      //.package(url: "https://github.com/wvabrinskas/NumSwift.git", branch: "main"),
      .package(url: "https://github.com/wvabrinskas/Logger.git", from: "1.0.6")
      //.package(url: "https://github.com/apple/swift-docc-plugin", branch: "main")
    ],
    targets: [
        .macro(
            name: "MacrosImpl",
            dependencies: [
                .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
                .product(name: "SwiftCompilerPlugin", package: "swift-syntax")
            ]
        ),
        .target(name: "MacrosDef", dependencies: ["MacrosImpl"]),
        .target(
            name: "Neuron",
            dependencies: [
              "NumSwift",
              "Logger",
              "MacrosDef"
            ],
            resources: [ .process("Resources") ]),
        .testTarget(
            name: "NeuronTests",
            dependencies: ["Neuron"]),
    ]
)
