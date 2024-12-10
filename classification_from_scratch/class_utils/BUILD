load("@aspect_rules_py//py:defs.bzl", "py_library")

py_library(
    name = "layers",
    srcs = [
        "__init__.py",
        "layers.py",
    ],
    visibility = ["//:__subpackages__"],
    deps = [
        "@pip//absl_py",
        "@pip//keras",
        "@pip//matplotlib",
        "@pip//numpy",
        "@pip//pillow",
        "@pip//pydot",
        "@pip//tensorflow",
    ],
)
