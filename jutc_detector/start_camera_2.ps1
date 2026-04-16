param(
    [ValidateSet("all", "recorder", "processor")]
    [string]$Mode = "all"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $scriptDir "start_jutc_instance.ps1") -EnvFile "camera_2.env" -Mode $Mode
