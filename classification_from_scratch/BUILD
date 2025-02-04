load("@rules_python//python:python.bzl", "py_binary")
load("@s_pip_deps//:requirements.bzl", "requirement", "data_requirement")
load("//bazel:zip_extract.bzl", "zip_extract")

# bazel query @bazel_tools//src/conditions:all -> vedi tutti i sistemi operativi che puoi usare in select
zip_extract(
    name = "cats_and_dogs",
    output_dir = "extracted_files",
    platform_flag = select({
        "//conditions:default": "unix",
        "@bazel_tools//src/conditions:windows": "windows",
    }),
    zip_file = "@cats_and_dogs//file",
)

py_binary(
    name = "classification_from_scratch",
    srcs = ["__main__.py"],
    main = "__main__.py",
    env = select({
        "@bazel_tools//src/conditions:windows": {
            "BAZEL_FIX_DLL": "Y"
        },
        "//conditions:default": {}
    }),
    visibility = ["//:__subpackages__"],
    deps = [
        "//classification_from_scratch/class_utils:layers",
        "//utils",
        requirement("absl_py"),
        requirement("keras"),
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("pydot"),
        requirement("pyqt6"),
        requirement("pyqt6-qt6"),
        requirement("pyqt6-sip"),
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
            requirement("tensorflow-intel"),
        ],
        "//conditions:default": []
    }),
    data = [
        ":cats_and_dogs"
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            data_requirement("tensorflow-intel"),
            data_requirement("pyqt6"),
            data_requirement("pyqt6-qt6"),
            data_requirement("pyqt6-sip"),
        ],
        "//conditions:default": []
    })
)
