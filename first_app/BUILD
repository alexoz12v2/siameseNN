load("@rules_python//python:python.bzl", "py_binary")
load("@pub//:requirements.bzl", "requirement", "data_requirement")

py_binary(
    name = "keras_test",
    srcs = ["__main__.py"],
    main = "__main__.py",
    deps = [
        "//first_app/apputils:layers",
        "//utils"
    ],
    env = select({
        "@bazel_tools//src/conditions:windows": {
            "BAZEL_FIX_DLL": "Y"
        },
        "//conditions:default": {}
    }),
)