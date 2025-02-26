def _zip_extract_impl(ctx):
    # Ensure the output directory exists
    output_dir = ctx.actions.declare_directory(ctx.attr.output_dir)

    # Determine the command based on the operating system
    if ctx.attr.platform_flag == "windows":
        new_name = ctx.file.zip_file.path[ctx.file.zip_file.path.rfind("/")+1:] + ".zip"
        old_name = new_name.rstrip(".zip")
        print("{new_name}".format(new_name=new_name))
        # For Windows, use the cmd command to unzip
        unzip_command = """
        /c/Windows/System32/WindowsPowerShell/v1.0/powershell -Command "
Rename-Item -Path \\"{zip_file}\\" -NewName \\"{new_name}\\";\\
Expand-Archive -Path '{zip_file}.zip' -DestinationPath '{output_dir}';\\
Rename-Item -Path \\"{zip_file}.zip\\" -NewName \\"{old_name}\\""
        """.format(
            old_name=old_name,
            new_name=new_name,
            output_dir=output_dir.path,
            zip_file=ctx.file.zip_file.path,
        )
    else:
        # For Unix-like systems (Linux/macOS), use `unzip` command
        unzip_command = """rm -rf {output_dir} && mkdir -p {output_dir} && unzip {zip_file} -d {output_dir}""".format(
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