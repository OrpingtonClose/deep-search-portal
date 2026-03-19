// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DeepSearch",
    platforms: [
        .iOS(.v17),
    ],
    products: [
        .library(
            name: "DeepSearch",
            targets: ["DeepSearch"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "DeepSearch",
            path: "DeepSearch"
        ),
    ]
)
