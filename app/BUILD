load("@aspect_rules_py//py:defs.bzl", aspect_py_binary = "py_binary")

# argomento che puoi passare resolutions, che e' un dict str -> label
aspect_py_binary(
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

# wiundows only (requirement lo prendi dal workspace creato da pip_parse)
py_binary(
    name = "keras_test_win",
    srcs = ["__main__.py"],
    main = "__main__.py",
    visibility = ["//visibility:public"],
    deps = [
        "//app/utils:layers_win",
    ],
)

load("@pip//:requirements.bzl", "requirement", "data_requirement")
load("@rules_pyvenv//:venv.bzl", "py_venv")
py_venv(
    name = "win_venv",
    deps = [
        "//app/utils:layers_win",
        requirement("absl-py"),
        requirement("tensorflow_intel"),
        requirement("keras"),
        requirement("tensorflow_directml_plugin"),
    ],
    venv_location = ".app+keras_test.venv",
)