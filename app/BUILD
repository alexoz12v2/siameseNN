load("@aspect_rules_py//py:defs.bzl", "py_binary")

py_binary(
    name = "keras_test",
    srcs = ["__main__.py"],
    main = "__main__.py",
    visibility = ["//:__subpackages__"],
    deps = [
        "//app/utils:layers",
        "@pip//absl_py",
        "@pip//keras",
        "@pip//tensorflow",
    ],
)
