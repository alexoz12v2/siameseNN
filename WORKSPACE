# Set the name of the bazel workspace.
workspace(name = "build_file_generation_example")

# Load the http_archive rule so that we can have bazel download
# various rulesets and dependencies.
# The `load` statement imports the symbol for http_archive from the http.bzl
# file.  When the symbol is loaded you can use the rule.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

######################################################################
# We need rules_go and bazel_gazelle, to build the gazelle plugin from source.
# Setup instructions for this section are at
# https://github.com/bazelbuild/bazel-gazelle#running-gazelle-with-bazel
# You may need to update the version of the rule, which is listed in the above
# documentation.
######################################################################

# Define an http_archive rule that will download the below ruleset,
# test the sha, and extract the ruleset to you local bazel cache.

http_archive(
    name = "io_bazel_rules_go",
    sha256 = "278b7ff5a826f3dc10f04feaf0b70d48b68748ccd512d7f98bf442077f043fe3",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.41.0/rules_go-v0.41.0.zip",
        "https://github.com/bazelbuild/rules_go/releases/download/v0.41.0/rules_go-v0.41.0.zip",
    ],
)

# Download the bazel_gazelle ruleset.
http_archive(
    name = "bazel_gazelle",
    sha256 = "d3fa66a39028e97d76f9e2db8f1b0c11c099e8e01bf363a923074784e451f809",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-gazelle/releases/download/v0.33.0/bazel-gazelle-v0.33.0.tar.gz",
        "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.33.0/bazel-gazelle-v0.33.0.tar.gz",
    ],
)

# Load rules_go ruleset and expose the toolchain and dep rules.
load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")
load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

# go_rules_dependencies is a function that registers external dependencies
# needed by the Go rules.
# See: https://github.com/bazelbuild/rules_go/blob/master/go/dependencies.rst#go_rules_dependencies
go_rules_dependencies()

# go_rules_dependencies is a function that registers external dependencies
# needed by the Go rules.
# See: https://github.com/bazelbuild/rules_go/blob/master/go/dependencies.rst#go_rules_dependencies
go_register_toolchains(version = "1.19.4")

# The following call configured the gazelle dependencies, Go environment and Go SDK.
gazelle_dependencies()

# Remaining setup is for rules_python.
http_archive(
    name = "rules_python",
    sha256 = "4f7e2aa1eb9aa722d96498f5ef514f426c1f55161c3c9ae628c857a7128ceb07",
    strip_prefix = "rules_python-1.0.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/1.0.0/rules_python-1.0.0.tar.gz",
)
http_archive(
  name = "rules_python_gazelle_plugin",
  sha256 = "4f7e2aa1eb9aa722d96498f5ef514f426c1f55161c3c9ae628c857a7128ceb07",
  strip_prefix = "rules_python-1.0.0/gazelle",
  url = "https://github.com/bazelbuild/rules_python/releases/download/1.0.0/rules_python-1.0.0.tar.gz",
)

http_archive(
    name = "aspect_rules_py",
    sha256 = "2ce48e0f3eaaf73204b623f99f23d45690b862a994b5b3c2464a2e361b0fc4ae",
    strip_prefix = "rules_py-1.0.0",
    url = "https://github.com/aspect-build/rules_py/releases/download/v1.0.0/rules_py-v1.0.0.tar.gz",
)

# Fetches the rules_py dependencies.
# If you want to have a different version of some dependency,
# you should fetch it *before* calling this.
# Alternatively, you can skip calling this function, so long as you've
# already fetched all the dependencies.
load("@aspect_rules_py//py:repositories.bzl", "rules_py_dependencies")

rules_py_dependencies()

load("@aspect_rules_py//py:toolchains.bzl", "rules_py_toolchains")

rules_py_toolchains()

# Next we load the setup and toolchain from rules_python.
load("@rules_python//python:repositories.bzl", "py_repositories", "python_register_toolchains")

# Perform general setup
py_repositories()

# We now register a hermetic Python interpreter rather than relying on a system-installed interpreter.
# This toolchain will allow bazel to download a specific python version, and use that version
# for compilation.
python_register_toolchains(
    name = "python39",
    python_version = "3.9",
)

load("@rules_python//python:pip.bzl", "pip_parse")

# This macro wraps the `pip_repository` rule that invokes `pip`, with `incremental` set.
# Accepts a locked/compiled requirements file and installs the dependencies listed within.
# Those dependencies become available in a generated `requirements.bzl` file.
# You can instead check this `requirements.bzl` file into your repo.
pip_parse(
    name = "pip",

    # Requirement groups allow Bazel to tolerate PyPi cycles by putting dependencies
    # which are known to form cycles into groups together.
    experimental_requirement_cycles = {
        "sphinx": [
            "sphinx",
            "sphinxcontrib-qthelp",
            "sphinxcontrib-htmlhelp",
            "sphinxcontrib-devhelp",
            "sphinxcontrib-applehelp",
            "sphinxcontrib-serializinghtml",
        ],
    },
    # (Optional) You can provide a python_interpreter (path) or a python_interpreter_target (a Bazel target, that
    # acts as an executable). The latter can be anything that could be used as Python interpreter. E.g.:
    # 1. Python interpreter that you compile in the build file.
    # 2. Pre-compiled python interpreter included with http_archive.
    # 3. Wrapper script, like in the autodetecting python toolchain.
    #
    # Here, we use the interpreter constant that resolves to the host interpreter from the default Python toolchain.
    python_interpreter_target = "@python39_host//:python",
    # Set the location of the lock file.
    requirements_lock = "//:requirements_lock.txt",
    requirements_windows = "//:requirements_windows.txt",
)

# Load the install_deps macro.
load("@pip//:requirements.bzl", "install_deps")

# Initialize repositories for all packages in requirements_lock.txt.
install_deps()

# The rules_python gazelle extension has some third-party go dependencies
# which we need to fetch in order to compile it.
load("@rules_python_gazelle_plugin//:deps.bzl", _py_gazelle_deps = "gazelle_deps")

# See: https://github.com/bazelbuild/rules_python/blob/main/gazelle/README.md
# This rule loads and compiles various go dependencies that running gazelle
# for python requirements.
_py_gazelle_deps()

# download e unzip di rules_pkg al fine di poter impacchettare i binary python in un archivio
http_archive(
    name = "rules_pkg",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/1.0.1/rules_pkg-1.0.1.tar.gz",
        "https://github.com/bazelbuild/rules_pkg/releases/download/1.0.1/rules_pkg-1.0.1.tar.gz",
    ],
    sha256 = "d20c951960ed77cb7b341c2a59488534e494d5ad1d30c4818c736d57772a9fef",
)
load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")
rules_pkg_dependencies()

# Download datasets cats and dogs
# generera nella cartella ~/.cache/bazel/_bazel_{USER}/{WORKSPACE dir hash}/execroot/{workspace name}/external/cats_and_dogs (essenzialmente la cartella del bazel sandbox)
# un WORKSPACE + una cartella "file" che contiene lo zip
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")
http_file(
    name = "cats_and_dogs",
    url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip",
)

# dataset per siamese networks: left => dataset anchors, right => dataset positive per ogni anchor
# sto usando due http_file targets, che creano un filegroup di un solo zip, piuttosto che una sola
# http_file che crea un filegroup di 2 zips, perche la mia rule custom zip_extract gestisce labels
# che si riferiscono ad un filegroup con un solo file
# quello che http_file fa e' una GET all'url che gli hai specificato
http_file(
    name = "siamese_left",
    url = "https://drive.usercontent.google.com/download?id=1jvkbTr_giSP3Ru8OwGNCg6B4PvVbcO34&authuser=0&confirm=t&uuid=9cf14af2-9021-4aa1-ae66-707c34a5b2c7&at=APvzH3r0KuHiI11CxX78ZPJk8pB1:1733746276346",
)

http_file(
    name = "siamese_right",
    url = "https://drive.usercontent.google.com/download?id=1EzBZUb_mh_Dp_FKD0P4XiYYSd0QBH5zW&authuser=0&confirm=t&uuid=0a26a7b6-8d05-4d19-ba60-112b2db37dee&at=APvzH3qj0KXtwq86j3DM9fiWyb-7:1733746709914",
)


