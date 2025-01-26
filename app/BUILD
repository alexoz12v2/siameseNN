load("@rules_python//python:python.bzl", "py_binary")
load("@s_pip_deps//:requirements.bzl", "requirement", "data_requirement")

py_binary(
    name = "keras_test",
    srcs = ["__main__.py"],
    main = "__main__.py",
    deps = [
        "//app/utils:layers",
        requirement("absl_py"),
        requirement("keras"),
    ] + select({
        "@bazel_tools//src/conditions:linux": [
            requirement("tensorflow[and-cuda]") # testare se funziona
            # requirement("tensorflow"),
            # requirement("nvidia-cublas-cu12")
            # requirement("nvidia-cuda-cupti-cu12")
            # requirement("nvidia-cuda-nvrtc-cu12")
            # requirement("nvidia-cuda-runtime-cu12")
            # requirement("nvidia-cudnn-cu12")
            # requirement("nvidia-cufft-cu12")
            # requirement("nvidia-curand-cu12" )
            # requirement("nvidia-cusolver-cu12")
            # requirement("nvidia-cusparse-cu12")
            # requirement("nvidia-nccl-cu12")
            # requirement("nvidia-nvjitlink-cu12" )
        ],
        "@bazel_tools//src/conditions:windows": [
            requirement("tensorflow-cpu"),
            requirement("tensorflow-directml-plugin")
        ],
        "//conditions:default": []
    }),
    data = [] + select({
        "@bazel_tools//src/conditions:windows": [
            data_requirement("tensorflow-directml-plugin")
        ],
        "//conditions:default": []
    })
)