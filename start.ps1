[CmdletBinding()]
param (
    [Parameter(Mandatory = $true)]
    [ValidateSet("siamese_first", "siamese_second", "classification_from_scratch", "first_app", IgnoreCase = $false)]
    [string]$App,

    [Parameter(Mandatory = $false)]
    [string[]]$ArgumentList = @()
)

Function Move-PythonFiles {
    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true)]
        [ValidateSet("siamese_first", "siamese_second", "classification_from_scratch", "first_app", IgnoreCase = $false)]
        [string]$App,

        [Parameter(Mandatory = $false)]
        [string[]]$ArgumentList = @()
    )

    # Get the script directory
    $ScriptDir = $PSScriptRoot
    $AppExe = Join-Path $ScriptDir "$App.exe"
    $AppPath = Join-Path $ScriptDir "$App.exe.runfiles"

    # Check if both the executable and its runfiles directory exist
    if (-Not (Test-Path $AppExe) -or -Not (Test-Path $AppPath)) {
        Write-Error "Executable or runfiles directory not found for: $App"
        
        Write-Output "Available applications:"
        @("siamese_first", "siamese_second", "classification_from_scratch", "first_app") | ForEach-Object {
            $ExePath = Join-Path $ScriptDir "$_.exe"
            $RunfilesPath = Join-Path $ScriptDir "$_.exe.runfiles"
            if ((Test-Path $ExePath) -and (Test-Path $RunfilesPath)) {
                Write-Output $_
            }
        }
        return
    }

    # Path to track whether the move operation has already been performed
    $StateFile = Join-Path $ScriptDir "move_state.json"

    # Load or initialize the state tracking file
    $State = @{}
    if (Test-Path $StateFile) {
        $StateObject = Get-Content $StateFile | ConvertFrom-Json
        $StateObject.psobject.properties | ForEach-Object { $State[$_.Name] = $_.Value }
    }

    # Check if the move operation is needed
    if (-Not $State.ContainsKey($App) -or -Not $State[$App]) {
        # Move directories matching "rules_python" from _main to the parent directory
        $MainPath = Join-Path $AppPath "_main"
        if (Test-Path $MainPath) {
            Get-ChildItem -Path $MainPath -Filter "rules_python*" | ForEach-Object {
                Move-Item -Path $_.FullName -Destination $AppPath -Force
            }
        }

        # Update state and save
        $State[$App] = $true
        $State | ConvertTo-Json | Set-Content $StateFile
    } else {
        # Verify that the files are correctly placed
        if ((Get-ChildItem -Path $AppPath -Filter "rules_python*").Count -eq 0) {
            Write-Error "Expected 'rules_python' files missing in $AppPath"
            return
        }
        if ((Get-ChildItem -Path (Join-Path $AppPath "_main") -Filter "rules_python*").Count -ne 0) {
            Write-Error "'rules_python' files should not be in _main but were found!"
            return
        }
    }

    Write-Output "Move operation completed successfully for $App."

    if ($ArgumentList.Count -eq 0) {
        Write-Output "Command Line: `"$($AppExe)`""
        $Proc = Start-Process -PassThru -Wait -NoNewWindow -FilePath $AppExe
    } else {
        Write-Output "Command Line: `"$($AppExe) $($ArgumentList -join " ")`""
        $Proc = Start-Process -PassThru -Wait -NoNewWindow -FilePath $AppExe -ArgumentList $ArgumentList
    }

    return $Proc.ExitCode
}

# If script is executed directly, call the function
Move-PythonFiles -App $App -ArgumentList $ArgumentList
