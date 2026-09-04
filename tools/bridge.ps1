# The bridge, launched the way this machine actually works.
#
#     .\tools\bridge.ps1
#
# There is a tools/bridge.sh beside this and it is the wrong shell here. The
# project runs on Windows and its dependencies are installed in the Windows
# interpreter; `bash tools/bridge.sh` from cmd opened WSL, whose Linux python
# has none of them, and the failure read as "No module named 'psycopg'" -- a
# missing package on a machine where the package is installed. Keep the sh
# version for a Linux host; use this one here.
#
# The address comes from .env, which the panel writes at the end of every
# successful setup. RunPod hands out a new one on every migration, so a bridge
# started from a remembered address points at a pod that no longer exists, and
# that failure looks like a broken pod rather than a stale address.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".env")) {
    Write-Error "no .env here. Run the panel first, or write POD_SSH into it."
    exit 1
}

# Read .env the way a shell would: skip comments and blanks, strip the quotes
# the panel writes around the address, and ignore anything that is not a
# NAME=value line. The file is edited by hand as well as by the panel.
Get-Content ".env" | ForEach-Object {
    $line = $_.Trim()
    if ($line -eq "" -or $line.StartsWith("#")) { return }
    $i = $line.IndexOf("=")
    if ($i -lt 1) { return }
    $name = $line.Substring(0, $i).Trim()
    if ($name -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') { return }
    $value = $line.Substring($i + 1).Trim()
    if ($value.Length -ge 2 -and $value.StartsWith('"') -and $value.EndsWith('"')) {
        $value = $value.Substring(1, $value.Length - 2)
    }
    Set-Item -Path "env:$name" -Value $value
}

if (-not $env:POD_SSH) {
    Write-Error ("POD_SSH is not set in .env. The panel writes it when a pod " +
                 'is ready; otherwise it looks like: POD_SSH="root@1.2.3.4 -p 22022"')
    exit 1
}

function Set-Default($name, $value) {
    if (-not (Get-Item -Path "env:$name" -ErrorAction SilentlyContinue)) {
        Set-Item -Path "env:$name" -Value $value
    }
}

Set-Default "DATABASE_URL" "postgresql://lookzi:lookzi@127.0.0.1:5433/lookzi"
Set-Default "S3_ENDPOINT" "http://127.0.0.1:9000"
Set-Default "S3_PUBLIC_ENDPOINT" "http://127.0.0.1:9000"
Set-Default "S3_KEY" "lookzi"
Set-Default "S3_SECRET" "lookzi-dev-secret"
Set-Default "S3_BUCKET" "lookzi"
# Every tool the product offers. A tool missing from this list is a tool whose
# jobs sit in the queue for ever while the studio says a worker is ready -- and
# nothing anywhere reports an error. short-video was missing from the sh
# version for exactly as long as it took to notice a clip that never started.
Set-Default "WORKER_TOOLS" ("product-to-model,virtual-try-on,model-swap," +
                            "packshot,model-creation,product-in-scene,short-video")

# The interpreter that has the dependencies, not the first one on PATH. Both
# exist on this machine and only one of them can import psycopg, so the
# candidate is tested rather than assumed.
$candidates = @()
if ($env:PYTHON) { $candidates += $env:PYTHON }
$candidates += @("D:\Python312\python.exe", "C:\Python312\python.exe")
$onPath = (Get-Command python -ErrorAction SilentlyContinue)
if ($onPath) { $candidates += $onPath.Source }
$onPath3 = (Get-Command python3 -ErrorAction SilentlyContinue)
if ($onPath3) { $candidates += $onPath3.Source }

$python = $null
foreach ($c in $candidates) {
    if (-not $c) { continue }
    if (-not (Test-Path $c)) { continue }
    & $c -c "import psycopg, boto3" 2>$null
    if ($LASTEXITCODE -eq 0) { $python = $c; break }
}

if (-not $python) {
    Write-Error ("no interpreter here can import psycopg and boto3. Install " +
                 "them with:  D:\Python312\python.exe -m pip install psycopg[binary] boto3" +
                 "  -- or set PYTHON to the interpreter that has them.")
    exit 1
}

# Only one at a time. Two bridges claim the same jobs and fight over the same
# forwarded port, and the symptom is jobs that intermittently fail rather than
# anything that names the cause.
$running = Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
    Where-Object { $_.CommandLine -match "tunnel_worker" }
if ($running) {
    Write-Host "a bridge is already running (pid $($running.ProcessId))."
    Write-Host "stop it first:  Stop-Process -Id $($running.ProcessId)"
    exit 1
}

Write-Host "bridge -> $env:POD_SSH"
Write-Host "python  -> $python"
& $python -m service.tunnel_worker
