load("@aspect_rules_py//py:defs.bzl", aspect_py_library="py_library")

aspect_py_library(
    name = "layers",
    srcs = [
        "__init__.py",
        "layers.py",
        "utils.py",
    ],
    visibility = ["//app:__pkg__"],
    deps = [
        "@pip//absl_py",
        "@pip//keras",
        "@pip//tensorflow",
    ],
)

py_library(
    name = "layers_win",
    srcs = [
        "__init__.py",
        "layers.py",
        "utils.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@pip//absl_py",
        "@pip//keras",
        "@pip//tensorflow_intel",
        "@pip//tensorflow_directml_plugin",
    ],
)