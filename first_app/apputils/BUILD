load("@rules_python//python:python.bzl", "py_library")
load("@pub//:requirements.bzl", "requirement", "data_requirement")

py_library(
    name = "layers",
    visibility = ["//visibility:public"],
    srcs = [
        ":__init__.py",
        ":layers.py",
        ":keras_utils.py",
    ],
    deps = [
        "//utils",
        requirement("absl_py"),
        requirement("keras"),
        requirement("numpy"),
    ] + select({
        "@bazel_tools//src/conditions:linux": [
            requirement("tensorflow"),
        ],
        "@bazel_tools//src/conditions:windows": [
            requirement("tensorflow-gpu"),
            requirement("nvidia-cublas-cu11"),
            requirement("nvidia-cuda-nvrtc-cu11"),
            requirement("nvidia-cuda-runtime-cu11"),
            requirement("nvidia-cudnn-cu11"),
            requirement("nvidia-cufft-cu11"),
            requirement("nvidia-cusparse-cu11"),
        ],
        "//conditions:default": []
    }),
)