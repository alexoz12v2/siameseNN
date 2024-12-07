load("@aspect_rules_py//py:defs.bzl", "py_library")

py_library(
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
