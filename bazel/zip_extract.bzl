# load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")
load("@rules_python//python:py_binary.bzl", "py_binary")


def _zip_extract_impl(ctx):
    # Ensure the output directory exists
    output_dir = ctx.actions.declare_directory(ctx.attr.output_dir)

    # Determine the command based on the operating system
    if ctx.attr.platform_flag == "windows":
        # For Windows, use the cmd command to unzip
        unzip_command = """
        mkdir "{output_dir}" && 
        powershell -Command "Expand-Archive -Path '{zip_file}' -DestinationPath '{output_dir}'"
        """.format(
            output_dir=output_dir.path,
            zip_file=ctx.file.zip_file.path,
        )
    else:
        # For Unix-like systems (Linux/macOS), use `unzip` command
        unzip_command = """
        rm -rf {output_dir} &&
        mkdir -p {output_dir} &&
        unzip {zip_file} -d {output_dir}
        """.format(
            output_dir=output_dir.path,
            zip_file=ctx.file.zip_file.path,
        )

    print("[{target}] running command {command}".format(target=ctx.label, command=unzip_command))
    
    # Extract the ZIP file using a spawn action
    ctx.actions.run_shell(
        inputs=[ctx.file.zip_file],
        outputs=[output_dir],
        command = unzip_command,
    )

    return [DefaultInfo(files=depset([output_dir]))]


zip_extract = rule(
    implementation=_zip_extract_impl,
    attrs={
        "zip_file": attr.label(allow_single_file=True, mandatory=True),  # Input ZIP file
        "platform_flag": attr.string(mandatory=True),
        "output_dir": attr.string(mandatory=True),  # Output directory
    },
)