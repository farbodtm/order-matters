package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//resnet/...",
    ],
)

filegroup(
    name = "py_srcs",
    data = glob([
        "**/*.py",
    ]),
)

py_library(
    name = "resnet_model",
    srcs = ["resnet_model.py"],
)

py_binary(
    name = "resnet_main",
    srcs = [
        "resnet_main.py",
    ],
    deps = [
        ":cifar_input",
        ":resnet_model",
    ],
)

py_library(
    name = "cifar_input",
    srcs = ["cifar_input.py"],
)

py_library(
    name = "order_model",
    srcs = ["order_model.py"],
)

py_binary(
    name = "order_main",
    srcs = [
        "order_main.py",
    ],
    deps = [
        ":cifar_input",
        ":order_model",
    ],
)

py_binary(
    name = "predict",
    srcs = [
        "predict.py",
    ],
    deps = [
        ":cifar_input",
        ":resnet_model",
    ],
)
