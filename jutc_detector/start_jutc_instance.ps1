param(
    [Parameter(Mandatory = $true)]
    [string]$EnvFile,

    [ValidateSet("all", "recorder", "processor")]
    [string]$Mode = "all"
)

$ErrorActionPreference = "Stop"

function Import-EnvFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Env file not found: $Path"
    }

    Get-Content -LiteralPath $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            return
        }

        $separatorIndex = $line.IndexOf("=")
        if ($separatorIndex -lt 1) {
            return
        }

        $name = $line.Substring(0, $separatorIndex).Trim()
        $value = $line.Substring($separatorIndex + 1).Trim()

        if (
            ($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))
        ) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Split-Path -Parent $scriptDir
$resolvedEnvFile = if ([System.IO.Path]::IsPathRooted($EnvFile)) {
    $EnvFile
} else {
    Join-Path $scriptDir $EnvFile
}

Import-EnvFile -Path $resolvedEnvFile

$existingPythonPath = [System.Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
if ([string]::IsNullOrWhiteSpace($existingPythonPath)) {
    $pythonPath = $projectDir
} else {
    $pythonPath = "$projectDir;$existingPythonPath"
}
[System.Environment]::SetEnvironmentVariable("PYTHONPATH", $pythonPath, "Process")

$cameraId = [System.Environment]::GetEnvironmentVariable("JUTC_CAMERA_ID", "Process")
$streamName = [System.Environment]::GetEnvironmentVariable("JUTC_STREAM_NAME", "Process")
$streamUrl = [System.Environment]::GetEnvironmentVariable("JUTC_STREAM_URL", "Process")

Write-Host "Starting JUTC detector instance"
Write-Host "  camera_id   = $cameraId"
Write-Host "  stream_name = $streamName"
Write-Host "  mode        = $Mode"
Write-Host "  env_file    = $resolvedEnvFile"
if ([string]::IsNullOrWhiteSpace($streamUrl)) {
    Write-Warning "JUTC_STREAM_URL is empty in $resolvedEnvFile"
}

Push-Location $projectDir
try {
    & py -3 -m jutc_detector.detector_service --mode $Mode
} finally {
    Pop-Location
}
