load("@rules_python//python:python.bzl", "py_binary")
load("@pub//:requirements.bzl", "requirement", "data_requirement")
load("//bazel:zip_extract.bzl", "zip_extract")

zip_extract(
    name = "siamese_left",
    output_dir = "siamese_left",
    platform_flag = select({
        "//conditions:default": "unix",
        "@bazel_tools//src/conditions:windows": "windows",
    }),
    zip_file = "@siamese_left//file",
)

zip_extract(
    name = "siamese_right",
    output_dir = "siamese_right",
    platform_flag = select({
        "//conditions:default": "unix",
        "@bazel_tools//src/conditions:windows": "windows",
    }),
    zip_file = "@siamese_right//file",
)

py_binary(
    name = "siamese_first",
    srcs = ["__main__.py"],
    main = "__main__.py",
    visibility = ["//:__subpackages__"],
    env = select({
        "@bazel_tools//src/conditions:windows": {
            "BAZEL_FIX_DLL": "Y"
        },
        "//conditions:default": {}
    }),
    deps = [
        "//siamese_first/siamese_lib",
        requirement("absl_py"),
        requirement("keras"),
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("pydot"),
        requirement("pyqt6"),
    ] + select({
        "@bazel_tools//src/conditions:linux": [
            requirement("tensorflow[and-cuda]") # testare se funziona
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
    data = [
        ":siamese_left",
        ":siamese_right",
    ]
)
