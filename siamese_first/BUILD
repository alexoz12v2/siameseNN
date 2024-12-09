load("@aspect_rules_py//py:defs.bzl", "py_binary")
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
    data = [
        ":siamese_left",
        ":siamese_right",
    ],
    main = "__main__.py",
    package_collisions = "warning",  # qt6 genera symlinks conflittuali, non mi interessa
    visibility = ["//:__subpackages__"],
    deps = [
        "//siamese_first/siamese_lib",
        "@pip//absl_py",
        "@pip//keras",
        "@pip//matplotlib",
        "@pip//numpy",
        "@pip//pydot",
        "@pip//pyqt6",
        "@pip//tensorflow",
    ],
)
