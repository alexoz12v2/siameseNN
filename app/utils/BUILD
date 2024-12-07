load("@aspect_rules_py//py:defs.bzl", "py_library")

py_library(
  name = "layers",
  srcs = [ "__init__.py", "layers.py", "utils.py" ],
  deps = [
    "@pip//absl_py",
    "@pip//keras",
    "@pip//tensorflow",
  ],
  visibility = ["//app:__pkg__"],
)