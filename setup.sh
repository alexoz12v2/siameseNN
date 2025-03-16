#!/bin/bash

# Get the absolute path of the current script's directory
CURRENT_DIR="$(dirname "$(realpath "$0")")"

# Path to the workspace_dir.bzl file
BZL_FILE="$CURRENT_DIR/bazel/workspace_dir.bzl"

# Ensure the file exists
if [[ ! -f "$BZL_FILE" ]]; then
    echo "Error: $BZL_FILE does not exist."
    exit 1
fi

# Update the workspace_dir in workspace_dir.bzl
sed -i "s|^workspace_dir = \".*\"|workspace_dir = \"$CURRENT_DIR\"|" "$BZL_FILE"

echo "workspace_dir updated to: $CURRENT_DIR"
