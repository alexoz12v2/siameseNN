load("@rules_python//python:python.bzl", "py_binary")
load("@pub//:requirements.bzl", "requirement", "data_requirement")
load("//bazel:zip_extract.bzl", "zip_extract")

# bazel query @bazel_tools//src/conditions:all -> vedi tutti i sistemi operativi che puoi usare in select
zip_extract(
    name = "cats_and_dogs",
    output_dir = "extracted_files",
    platform_flag = select({
        "//conditions:default": "unix",
        "@bazel_tools//src/conditions:windows": "windows",
    }),
    zip_file = "@cats_and_dogs//file",
)

py_binary(
    name = "classification_from_scratch",
    srcs = ["__main__.py"],
    main = "__main__.py",
    visibility = ["//visibility:public"],
    env = select({
        "@bazel_tools//src/conditions:linux": {
            "LD_LIBRARY_PATH": 
                "/home/alessio/programming/siameseNN/bazel-bin/classification_from_scratch/classification_from_scratch.runfiles/rules_python~~pip~pub_310_pyqt6_qt6/site-packages/PyQt6/Qt6/lib" + ":" + 
                "/home/alessio/programming/siameseNN/bazel-bin/classification_from_scratch/classification_from_scratch.runfiles/rules_python~~pip~pub_310_nvidia_nvjitlink_cu12/site-packages/nvidia/nvjitlink/lib" + ":" + 
                "/home/alessio/programming/siameseNN/bazel-bin/classification_from_scratch/classification_from_scratch.runfiles/rules_python~~pip~pub_310_nvidia_nccl_cu12/site-packages/nvidia/nccl/lib" + ":" + 
                "/home/alessio/programming/siameseNN/bazel-bin/classification_from_scratch/classification_from_scratch.runfiles/rules_python~~pip~pub_310_nvidia_cusparse_cu12/site-packages/nvidia/cusparse/lib" + ":" + 
                "/home/alessio/programming/siameseNN/bazel-bin/classification_from_scratch/classification_from_scratch.runfiles/rules_python~~pip~pub_310_nvidia_cusolver_cu12/site-packages/nvidia/cusolver/lib" + ":" + 
                "/home/alessio/programming/siameseNN/bazel-bin/classification_from_scratch/classification_from_scratch.runfiles/rules_python~~pip~pub_310_nvidia_curand_cu12/site-packages/nvidia/curand/lib" + ":" + 
                "/home/alessio/programming/siameseNN/bazel-bin/classification_from_scratch/classification_from_scratch.runfiles/rules_python~~pip~pub_310_nvidia_cufft_cu12/site-packages/nvidia/cufft/lib" + ":" + 
                "/home/alessio/programming/siameseNN/bazel-bin/classification_from_scratch/classification_from_scratch.runfiles/rules_python~~pip~pub_310_nvidia_cudnn_cu12/site-packages/nvidia/cudnn/lib" + ":" + 
                "/home/alessio/programming/siameseNN/bazel-bin/classification_from_scratch/classification_from_scratch.runfiles/rules_python~~pip~pub_310_nvidia_cuda_runtime_cu12/site-packages/nvidia/cuda_runtime/lib" + ":" + 
                "/home/alessio/programming/siameseNN/bazel-bin/classification_from_scratch/classification_from_scratch.runfiles/rules_python~~pip~pub_310_nvidia_cuda_nvrtc_cu12/site-packages/nvidia/cuda_nvrtc/lib" + ":" + 
                "/home/alessio/programming/siameseNN/bazel-bin/classification_from_scratch/classification_from_scratch.runfiles/rules_python~~pip~pub_310_nvidia_cuda_cupti_cu12/site-packages/nvidia/cuda_cupti/lib" + ":" + 
                "/home/alessio/programming/siameseNN/bazel-bin/classification_from_scratch/classification_from_scratch.runfiles/rules_python~~pip~pub_310_nvidia_cublas_cu12/site-packages/nvidia/cublas/lib",
            "XLA_FLAGS": "--xla_gpu_cuda_data_dir=/home/alessio/programming/siameseNN/bazel-bin/classification_from_scratch/classification_from_scratch.runfiles/rules_python~~pip~pub_310_nvidia_cuda_nvcc_cu12/site-packages/nvidia/cuda_nvcc"
        },
        "//conditions:default": {}
    }),
    deps = [
        "//classification_from_scratch/class_utils:layers",
        "//utils",
        requirement("absl_py"),
        requirement("keras"),
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("pydot"),
        requirement("pyqt6"),
        requirement("pyqt6-qt6"),
        requirement("pyqt6-sip"),
    ] + select({
        "@bazel_tools//src/conditions:linux": [
            requirement("tensorflow"),
        ],
        "@bazel_tools//src/conditions:windows": [
            requirement("tensorflow-gpu"),
            requirement("nvidia-cublas-cu11"),
            requirement("nvidia-cuda-nvrtc-cu11"),
            requirement("nvidia-cuda-runtime-cu11"),
            requirement("nvidia-cudnn-cu11"),
            requirement("nvidia-cufft-cu11"),
            requirement("nvidia-cusparse-cu11"),
        ],
        "//conditions:default": []
    }),
    data = [
        ":cats_and_dogs"
    ] + select({
        "@bazel_tools//src/conditions:linux": [
            data_requirement("tensorflow"),
        ],
        "//conditions:default": []
    })
)