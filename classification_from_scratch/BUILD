load("@aspect_rules_py//py:defs.bzl", "py_binary")
load("//bazel:zip_extract.bzl", "zip_extract")

# bazel query @bazel_tools//src/conditions:all -> vedi tutti i sistemi operativi che puoi usare in select
zip_extract(
    name = "cats_and_dogs",
    zip_file = "@cats_and_dogs//file",
    output_dir = "extracted_files",
    platform_flag = select({
        "//conditions:default": "unix",
        "@bazel_tools//src/conditions:windows": "windows",
    }),
)

py_binary(
    name = "classification_from_scratch",
    srcs = ["__main__.py"],
    data = [":cats_and_dogs"],
    main = "__main__.py",
    visibility = ["//:__subpackages__"],
    deps = [
        "//classification_from_scratch/class_utils:layers",
        "@pip//absl_py",
        "@pip//keras",
        "@pip//tensorflow",
    ],
)
