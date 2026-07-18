param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$JsonOut = ""
)

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    $python = Get-Command py -ErrorAction SilentlyContinue
}

if (-not $python) {
    Write-Error "Python was not found on PATH. Install Python or open a shell where Python is available."
    exit 2
}

$script = Join-Path $PSScriptRoot "verify_cpp_build.py"
$argsList = @($script, "--repo-root", $RepoRoot)
if ($JsonOut) {
    $argsList += @("--json-out", $JsonOut)
}

& $python.Source @argsList
exit $LASTEXITCODE
