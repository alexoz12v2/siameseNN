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
    visibility = ["//visibility:public"],
    env = select({
        "@bazel_tools//src/conditions:windows": {
            "BAZEL_FIX_DLL": "Y"
        },
        "//conditions:default": {}
    }),
    deps = [
        "//utils",
        "//siamese_first/siamese_lib",
    ],
    data = [
        ":siamese_left",
        ":siamese_right",
    ]
)
