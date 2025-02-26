#!/bin/bash

# Get the directory of the script (which should be next to the executables)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# List of valid executables
VALID_EXECUTABLES=("keras_test" "classification_from_scratch" "siamese_first" "siamese_second")

# Ensure an executable is passed
if [ -z "$1" ]; then
    echo "Usage: $0 <executable_name> [-- additional arguments]"
    echo "Valid executables: ${VALID_EXECUTABLES[*]}"
    exit 1
fi

# Extract executable name
EXECUTABLE="$1"
shift  # Remove the executable name from arguments

# Capture additional arguments after "--"
EXTRA_ARGS=()
PASSTHROUGH=false
for arg in "$@"; do
    if [ "$PASSTHROUGH" = true ]; then
        EXTRA_ARGS+=("$arg")
    elif [ "$arg" == "--" ]; then
        PASSTHROUGH=true
    fi
done

RUNFILES_DIR="$SCRIPT_DIR/${EXECUTABLE}.runfiles"

# Check if the executable is valid
if [[ ! " ${VALID_EXECUTABLES[@]} " =~ " ${EXECUTABLE} " ]]; then
    echo "Error: Unrecognized executable '$EXECUTABLE'."
    echo "Valid executables: ${VALID_EXECUTABLES[*]}"
    exit 1
fi

# Validate the executable file exists and is executable
if [ ! -x "$SCRIPT_DIR/$EXECUTABLE" ]; then
    echo "Error: Executable '$EXECUTABLE' not found in $SCRIPT_DIR or is not executable."
    exit 1
fi

# Save the original environment variables
export OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export OLD_XLA_FLAGS="$XLA_FLAGS"

# Set up the environment
export LD_LIBRARY_PATH="$RUNFILES_DIR/rules_python~~pip~pub_310_pyqt6_qt6/site-packages/PyQt6/Qt6/lib:\
$RUNFILES_DIR/rules_python~~pip~pub_310_nvidia_nvjitlink_cu12/site-packages/nvidia/nvjitlink/lib:\
$RUNFILES_DIR/rules_python~~pip~pub_310_nvidia_nccl_cu12/site-packages/nvidia/nccl/lib:\
$RUNFILES_DIR/rules_python~~pip~pub_310_nvidia_cusparse_cu12/site-packages/nvidia/cusparse/lib:\
$RUNFILES_DIR/rules_python~~pip~pub_310_nvidia_cusolver_cu12/site-packages/nvidia/cusolver/lib:\
$RUNFILES_DIR/rules_python~~pip~pub_310_nvidia_curand_cu12/site-packages/nvidia/curand/lib:\
$RUNFILES_DIR/rules_python~~pip~pub_310_nvidia_cufft_cu12/site-packages/nvidia/cufft/lib:\
$RUNFILES_DIR/rules_python~~pip~pub_310_nvidia_cudnn_cu12/site-packages/nvidia/cudnn/lib:\
$RUNFILES_DIR/rules_python~~pip~pub_310_nvidia_cuda_runtime_cu12/site-packages/nvidia/cuda_runtime/lib:\
$RUNFILES_DIR/rules_python~~pip~pub_310_nvidia_cuda_nvrtc_cu12/site-packages/nvidia/cuda_nvrtc/lib:\
$RUNFILES_DIR/rules_python~~pip~pub_310_nvidia_cuda_cupti_cu12/site-packages/nvidia/cuda_cupti/lib:\
$RUNFILES_DIR/rules_python~~pip~pub_310_nvidia_cublas_cu12/site-packages/nvidia/cublas/lib"

export XLA_FLAGS="--xla_gpu_cuda_data_dir=$RUNFILES_DIR/rules_python~~pip~pub_310_nvidia_cuda_nvcc_cu12/site-packages/nvidia/cuda_nvcc"

# Run the executable with additional arguments
echo "Running $EXECUTABLE with configured environment..."
"$SCRIPT_DIR/$EXECUTABLE" "${EXTRA_ARGS[@]}"

# Restore the original environment
export LD_LIBRARY_PATH="$OLD_LD_LIBRARY_PATH"
export XLA_FLAGS="$OLD_XLA_FLAGS"

echo "Environment restored."
