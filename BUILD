# Load various rules so that we can have bazel download
# various rulesets and dependencies.
# The `load` statement imports the symbol for the rule, in the defined
# ruleset. When the symbol is loaded you can use the rule.
load("@bazel_gazelle//:def.bzl", "gazelle")
load("@pub//:requirements.bzl", "all_whl_requirements", "requirement")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
load("@rules_python_gazelle_plugin//manifest:defs.bzl", "gazelle_python_manifest")
load("@rules_python_gazelle_plugin//modules_mapping:def.bzl", "modules_mapping")
load("@rules_python_gazelle_plugin//:def.bzl", "GAZELLE_PYTHON_RUNTIME_DEPS")
load("@rules_pkg//pkg:zip.bzl", "pkg_zip")

compile_pip_requirements(
    name = "requirements",
    src = ":pyproject.toml", # default, ma meglio essere espliciti
    requirements_txt = "requirements_lock.txt",
    requirements_windows = "requirements_windows.txt",
    visibility = ["//visibility:public"],
)

# This repository rule fetches the metadata for python packages we
# depend on. That data is required for the gazelle_python_manifest
# rule to update our manifest file.
# To see what this rule does, try `bazel run @modules_map//:print`
modules_mapping(
    name = "modules_map",
    exclude_patterns = [
        "^_|(\\._)+",  # This is the default.
        "(\\.tests)+",  # Add a custom one to get rid of the psutil tests.
    ],
    wheels = all_whl_requirements,
)

# Gazelle python extension needs a manifest file mapping from
# an import to the installed package that provides it.
# This macro produces two targets:
# - //:gazelle_python_manifest.update can be used with `bazel run`
#   to recalculate the manifest
# - //:gazelle_python_manifest.test is a test target ensuring that
#   the manifest doesn't need to be updated
gazelle_python_manifest(
    name = "gazelle_python_manifest",
    modules_mapping = ":modules_map",
    pip_repository_name = "pub",
    requirements = "//:requirements_lock.txt",
)

# Our gazelle target points to the python gazelle binary.
# This is the simple case where we only need one language supported.
# If you also had proto, go, or other gazelle-supported languages,
# you would also need a gazelle_binary rule.
# See https://github.com/bazelbuild/bazel-gazelle/blob/master/extend.rst#example
# gazelle:prefix github.com/alexoz12v2/siameseNN
# non va su windows, non mi interessa, quello che mi importa e' che i py_binary vadano
gazelle(
    name = "gazelle",
    data = GAZELLE_PYTHON_RUNTIME_DEPS,
    gazelle = "@rules_python_gazelle_plugin//python:gazelle_binary",
)

# filegroup target per fare il packaging delle applicazioni python
# no. usare py_pex_binary
pkg_zip(
    name = "app_zip",
    visibility = ["//visibility:public"],
    srcs = [ 
        "//first_app:keras_test", 
        "//classification_from_scratch:classification_from_scratch",
        "//siamese_first:siamese_first",
        "//siamese_second",
    ] + select({
        "@bazel_tools//src/conditions:linux": [
            ":start.sh"
        ],
        "//conditions:default": []
    }),
    include_runfiles = True,
)

pkg_zip(
    name = "app_zip.light",
    visibility = [ "//visibility:public" ],
    srcs = [
        "//siamese_second",
    ] + select({
        "@bazel_tools//src/conditions:linux": [
            ":start.sh"
        ],
        "//conditions:default": []
    }),
    include_runfiles = True,
)