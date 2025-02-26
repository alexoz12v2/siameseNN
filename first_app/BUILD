load("@rules_python//python:python.bzl", "py_binary")
load("@pub//:requirements.bzl", "requirement", "data_requirement")

py_binary(
    name = "keras_test",
    srcs = ["__main__.py"],
    main = "__main__.py",
    visibility = ["//visibility:public"],
    env = select({
        "@bazel_tools//src/conditions:linux": {
            "LD_LIBRARY_PATH": 
                "/home/alessio/programming/siameseNN/bazel-bin/first_app/keras_test.runfiles/rules_python~~pip~pub_310_pyqt6_qt6/site-packages/PyQt6/Qt6/lib" + ":" + 
                "/home/alessio/programming/siameseNN/bazel-bin/first_app/keras_test.runfiles/rules_python~~pip~pub_310_nvidia_nvjitlink_cu12/site-packages/nvidia/nvjitlink/lib" + ":" + 
                "/home/alessio/programming/siameseNN/bazel-bin/first_app/keras_test.runfiles/rules_python~~pip~pub_310_nvidia_nccl_cu12/site-packages/nvidia/nccl/lib" + ":" + 
                "/home/alessio/programming/siameseNN/bazel-bin/first_app/keras_test.runfiles/rules_python~~pip~pub_310_nvidia_cusparse_cu12/site-packages/nvidia/cusparse/lib" + ":" + 
                "/home/alessio/programming/siameseNN/bazel-bin/first_app/keras_test.runfiles/rules_python~~pip~pub_310_nvidia_cusolver_cu12/site-packages/nvidia/cusolver/lib" + ":" + 
                "/home/alessio/programming/siameseNN/bazel-bin/first_app/keras_test.runfiles/rules_python~~pip~pub_310_nvidia_curand_cu12/site-packages/nvidia/curand/lib" + ":" + 
                "/home/alessio/programming/siameseNN/bazel-bin/first_app/keras_test.runfiles/rules_python~~pip~pub_310_nvidia_cufft_cu12/site-packages/nvidia/cufft/lib" + ":" + 
                "/home/alessio/programming/siameseNN/bazel-bin/first_app/keras_test.runfiles/rules_python~~pip~pub_310_nvidia_cudnn_cu12/site-packages/nvidia/cudnn/lib" + ":" + 
                "/home/alessio/programming/siameseNN/bazel-bin/first_app/keras_test.runfiles/rules_python~~pip~pub_310_nvidia_cuda_runtime_cu12/site-packages/nvidia/cuda_runtime/lib" + ":" + 
                "/home/alessio/programming/siameseNN/bazel-bin/first_app/keras_test.runfiles/rules_python~~pip~pub_310_nvidia_cuda_nvrtc_cu12/site-packages/nvidia/cuda_nvrtc/lib" + ":" + 
                "/home/alessio/programming/siameseNN/bazel-bin/first_app/keras_test.runfiles/rules_python~~pip~pub_310_nvidia_cuda_cupti_cu12/site-packages/nvidia/cuda_cupti/lib" + ":" + 
                "/home/alessio/programming/siameseNN/bazel-bin/first_app/keras_test.runfiles/rules_python~~pip~pub_310_nvidia_cublas_cu12/site-packages/nvidia/cublas/lib",
            "XLA_FLAGS": "--xla_gpu_cuda_data_dir=/home/alessio/programming/siameseNN/bazel-bin/first_app/keras_test.runfiles/rules_python~~pip~pub_310_nvidia_cuda_nvcc_cu12/site-packages/nvidia/cuda_nvcc"
        },
        "//conditions:default": {}
    }),
    deps = [
        "//first_app/apputils:layers",
        "//utils"
    ],
)