param(
  [int]$Port = 5000,
  [string]$BindHost = "127.0.0.1",
  [string]$TunnelName,
  [string]$PublicHostname,
  [string]$TunnelTokenFile,
  [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding

$projectRoot = Split-Path -Parent $PSScriptRoot
$tmpDir = Join-Path $projectRoot "docs\.tmp"
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

$serverOutLog = Join-Path $tmpDir "server-tunnel.out.log"
$serverErrLog = Join-Path $tmpDir "server-tunnel.err.log"
$tunnelOutLog = Join-Path $tmpDir "cloudflared-tunnel.out.log"
$tunnelStdOutLog = Join-Path $tmpDir "cloudflared-tunnel.stdout.log"
$tunnelErrLog = Join-Path $tmpDir "cloudflared-tunnel.err.log"
$serverPidFile = Join-Path $tmpDir "server-tunnel.pid"
$tunnelPidFile = Join-Path $tmpDir "cloudflared-tunnel.pid"
$generatedTokenFile = Join-Path $tmpDir "cloudflared.named.token"

if (-not ("KillOnCloseJob.Native" -as [type])) {
  Add-Type -Language CSharp -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

namespace KillOnCloseJob {
  public static class Native {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern IntPtr CreateJobObject(IntPtr lpJobAttributes, string lpName);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool SetInformationJobObject(
      IntPtr hJob,
      JOBOBJECTINFOCLASS JobObjectInformationClass,
      IntPtr lpJobObjectInfo,
      uint cbJobObjectInfoLength
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool CloseHandle(IntPtr handle);

    public const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;

    public enum JOBOBJECTINFOCLASS {
      JobObjectExtendedLimitInformation = 9
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct IO_COUNTERS {
      public ulong ReadOperationCount;
      public ulong WriteOperationCount;
      public ulong OtherOperationCount;
      public ulong ReadTransferCount;
      public ulong WriteTransferCount;
      public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_BASIC_LIMIT_INFORMATION {
      public long PerProcessUserTimeLimit;
      public long PerJobUserTimeLimit;
      public uint LimitFlags;
      public UIntPtr MinimumWorkingSetSize;
      public UIntPtr MaximumWorkingSetSize;
      public uint ActiveProcessLimit;
      public UIntPtr Affinity;
      public uint PriorityClass;
      public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
      public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
      public IO_COUNTERS IoInfo;
      public UIntPtr ProcessMemoryLimit;
      public UIntPtr JobMemoryLimit;
      public UIntPtr PeakProcessMemoryUsed;
      public UIntPtr PeakJobMemoryUsed;
    }
  }
}
"@
}

function Write-Step {
  param([string]$Message)
  Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Message)
}

function Resolve-CommandPath {
  param([Parameter(Mandatory = $true)][string]$CommandName)
  $command = Get-Command $CommandName -ErrorAction SilentlyContinue
  if ($command) {
    return $command.Source
  }
  return $null
}

function Get-CloudflaredPath {
  $direct = Resolve-CommandPath -CommandName "cloudflared"
  if ($direct) {
    return $direct
  }

  $candidates = @(
    "$env:ProgramFiles\cloudflared\cloudflared.exe",
    "$env:USERPROFILE\AppData\Local\Microsoft\WinGet\Links\cloudflared.exe"
  )

  foreach ($candidate in $candidates) {
    if (Test-Path $candidate) {
      return $candidate
    }
  }

  $wingetMatch = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Filter "cloudflared.exe" -Recurse -ErrorAction SilentlyContinue |
    Select-Object -First 1

  if ($wingetMatch) {
    return $wingetMatch.FullName
  }

  throw "cloudflared executable was not found."
}

function Import-DotEnvFile {
  param([string]$Path)

  if (-not (Test-Path $Path)) {
    return
  }

  foreach ($line in Get-Content -Path $Path -ErrorAction SilentlyContinue) {
    $trimmed = $line.Trim()
    if (-not $trimmed -or $trimmed.StartsWith("#")) {
      continue
    }

    $pair = $trimmed -split "=", 2
    if ($pair.Count -ne 2) {
      continue
    }

    $key = $pair[0].Trim()
    $value = $pair[1].Trim()
    if ($value.StartsWith('"') -and $value.EndsWith('"')) {
      $value = $value.Substring(1, $value.Length - 2)
    } elseif ($value.StartsWith("'") -and $value.EndsWith("'")) {
      $value = $value.Substring(1, $value.Length - 2)
    }

    if (-not [string]::IsNullOrWhiteSpace($key) -and -not [Environment]::GetEnvironmentVariable($key, "Process")) {
      [Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
  }
}

function Get-SettingValue {
  param(
    [string[]]$Keys,
    [string]$Default = ""
  )

  foreach ($key in $Keys) {
    $value = [Environment]::GetEnvironmentVariable($key, "Process")
    if (-not [string]::IsNullOrWhiteSpace($value)) {
      return $value.Trim()
    }
  }

  return $Default
}

function New-KillOnCloseJobHandle {
  $jobHandle = [KillOnCloseJob.Native]::CreateJobObject([IntPtr]::Zero, $null)
  if ($jobHandle -eq [IntPtr]::Zero) {
    throw "Failed to create Windows Job Object."
  }

  $info = New-Object KillOnCloseJob.Native+JOBOBJECT_EXTENDED_LIMIT_INFORMATION
  $info.BasicLimitInformation.LimitFlags = [KillOnCloseJob.Native]::JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

  $size = [Runtime.InteropServices.Marshal]::SizeOf([type]([KillOnCloseJob.Native+JOBOBJECT_EXTENDED_LIMIT_INFORMATION]))
  $ptr = [Runtime.InteropServices.Marshal]::AllocHGlobal($size)

  try {
    [Runtime.InteropServices.Marshal]::StructureToPtr($info, $ptr, $false)
    $ok = [KillOnCloseJob.Native]::SetInformationJobObject(
      $jobHandle,
      [KillOnCloseJob.Native+JOBOBJECTINFOCLASS]::JobObjectExtendedLimitInformation,
      $ptr,
      [uint32]$size
    )

    if (-not $ok) {
      throw "Failed to configure Job Object."
    }
  }
  finally {
    [Runtime.InteropServices.Marshal]::FreeHGlobal($ptr)
  }

  return $jobHandle
}

function Add-ProcessToJob {
  param(
    [Parameter(Mandatory = $true)]$JobHandle,
    [Parameter(Mandatory = $true)][System.Diagnostics.Process]$Process
  )

  $null = $Process.Handle
  $ok = [KillOnCloseJob.Native]::AssignProcessToJobObject($JobHandle, $Process.Handle)
  if (-not $ok) {
    throw "Failed to assign process ID=$($Process.Id) to Job Object."
  }
}

function Stop-ProcessFromPidFile {
  param([string]$PidFile)

  if (-not (Test-Path $PidFile)) {
    return
  }

  $pidValue = (Get-Content -Path $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if (-not [string]::IsNullOrWhiteSpace($pidValue)) {
    try {
      Stop-Process -Id ([int]$pidValue) -Force -ErrorAction SilentlyContinue
    } catch {}
  }

  Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

function Stop-ExistingManagedProcesses {
  param(
    [int]$Port,
    [string]$TunnelName,
    [string]$TunnelLogPath,
    [string]$TokenFilePath
  )

  Stop-ProcessFromPidFile -PidFile $serverPidFile
  Stop-ProcessFromPidFile -PidFile $tunnelPidFile

  $listeningProcesses = @()
  try {
    $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop
    foreach ($listener in @($listeners)) {
      $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)" -ErrorAction SilentlyContinue
      if ($proc) {
        $listeningProcesses += $proc
      }
    }
  } catch {}

  $seenProcessIds = New-Object 'System.Collections.Generic.HashSet[int]'

  foreach ($proc in $listeningProcesses) {
    if ($seenProcessIds.Add([int]$proc.ProcessId)) {
      Write-Step ("포트 {0} 점유 프로세스 정리: PID={1} NAME={2}" -f $Port, $proc.ProcessId, $proc.Name)
      try {
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
      } catch {}
    }
  }

  $pythonTargets = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
      $_.Name -match '^python([0-9\.]+)?(\.exe)?$' -and
      $_.CommandLine -match 'app\.py' -and
      $_.CommandLine -match 'ncai-dementia-risk-monitor'
    }

  foreach ($proc in $pythonTargets) {
    if ($seenProcessIds.Add([int]$proc.ProcessId)) {
      Write-Step ("기존 Python 앱 정리: PID={0}" -f $proc.ProcessId)
      try {
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
      } catch {}
    }
  }

  $escapedTunnelName = [regex]::Escape($TunnelName)
  $escapedLogPath = [regex]::Escape($TunnelLogPath)
  $escapedTokenPath = [regex]::Escape($TokenFilePath)
  $tunnelTargets = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
      $_.Name -eq 'cloudflared.exe' -and
      $_.CommandLine -match 'tunnel' -and (
        $_.CommandLine -match $escapedTunnelName -or
        $_.CommandLine -match $escapedLogPath -or
        $_.CommandLine -match $escapedTokenPath
      )
    }

  foreach ($proc in $tunnelTargets) {
    if ($seenProcessIds.Add([int]$proc.ProcessId)) {
      Write-Step ("기존 Named Tunnel 정리: PID={0}" -f $proc.ProcessId)
      try {
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
      } catch {}
    }
  }
}

function Wait-ForHealth {
  param(
    [string]$Url,
    [int]$TimeoutSeconds = 90
  )

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    try {
      $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 5
      if ($response.StatusCode -eq 200) {
        return
      }
    } catch {}
    Start-Sleep -Milliseconds 1200
  }

  throw "Server health check failed: $Url"
}

function Get-HealthPayload {
  param([string]$Url)

  try {
    $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 5
    if ($response.StatusCode -eq 200 -and $response.Content) {
      return $response.Content | ConvertFrom-Json
    }
  } catch {}

  return $null
}

function Get-LanUrls {
  param([int]$Port)

  $urls = New-Object System.Collections.Generic.List[string]

  try {
    $configs = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop |
      Where-Object {
        $_.IPAddress -match '^\d+\.\d+\.\d+\.\d+$' -and
        $_.IPAddress -notlike '127.*' -and
        $_.PrefixOrigin -ne 'WellKnown'
      } |
      Sort-Object InterfaceMetric, InterfaceAlias, IPAddress

    foreach ($config in $configs) {
      $urls.Add(("http://{0}:{1}" -f $config.IPAddress, $Port))
    }
  } catch {}

  return @($urls | Select-Object -Unique)
}

function Wait-ForNamedTunnelReady {
  param(
    [string[]]$LogPaths,
    [System.Diagnostics.Process]$Process,
    [int]$TimeoutSeconds = 90
  )

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  $seenLogLines = New-Object 'System.Collections.Generic.HashSet[string]'
  $stabilityThreshold = (Get-Date).AddSeconds(8)

  while ((Get-Date) -lt $deadline) {
    if ($Process -and $Process.HasExited) {
      throw "Cloudflare Named Tunnel process exited before it became ready."
    }

    foreach ($logPath in $LogPaths) {
      if (-not (Test-Path $logPath)) {
        continue
      }

      try {
        $logLines = Get-Content -Path $logPath -ErrorAction SilentlyContinue
        if (-not $logLines) {
          continue
        }

        foreach ($line in ($logLines | Select-Object -Last 5)) {
          $normalizedLine = ($line | Out-String).Trim()
          if ($normalizedLine -and $seenLogLines.Add($normalizedLine)) {
            Write-Host ("      cloudflared > {0}" -f $normalizedLine)
          }
        }

        if ($logLines -match 'Registered tunnel connection' -or
            $logLines -match 'Connection [a-f0-9-]+ registered' -or
            $logLines -match 'Initial protocol') {
          return
        }

        if ($logLines -match 'ERR ' -and $logLines -notmatch 'timeout: no recent network activity') {
          $lastError = ($logLines | Select-String -Pattern 'ERR ' | Select-Object -Last 1).Line
          throw ("Cloudflare Named Tunnel error: {0}" -f $lastError)
        }
      } catch {
        if ($_.Exception.Message -like 'Cloudflare Named Tunnel error:*') {
          throw
        }
      }
    }

    if ((Get-Date) -ge $stabilityThreshold) {
      return
    }

    Start-Sleep -Milliseconds 1200
  }

  throw "Cloudflare Named Tunnel did not become ready in time."
}

function Get-TunnelTokenFilePath {
  param([string]$ProjectRoot)

  if ($TunnelTokenFile -and (Test-Path $TunnelTokenFile)) {
    return (Resolve-Path $TunnelTokenFile).Path
  }

  $candidateFiles = @(
    (Join-Path $ProjectRoot "data\cloudflared_tunnel_token.txt"),
    (Join-Path $ProjectRoot "data\cloudflare_tunnel_token.txt"),
    (Join-Path $ProjectRoot "data\tunnel_token.txt")
  )

  foreach ($candidate in $candidateFiles) {
    if (Test-Path $candidate) {
      return (Resolve-Path $candidate).Path
    }
  }

  $tokenValue = Get-SettingValue -Keys @(
    "CLOUDFLARE_TUNNEL_TOKEN",
    "CLOUDFLARED_TUNNEL_TOKEN",
    "TUNNEL_TOKEN"
  )

  if ($tokenValue) {
    [System.IO.File]::WriteAllText(
      $generatedTokenFile,
      $tokenValue,
      [System.Text.UTF8Encoding]::new($false)
    )
    return $generatedTokenFile
  }

  return $null
}

function Write-StatusBlock {
  param(
    [string]$BindHost,
    [int]$Port,
    [string[]]$LanUrls,
    [string]$PublicUrl,
    [string]$HealthUrl,
    $HealthPayload,
    [string]$TunnelName
  )

  Write-Host ""
  Write-Host "========================================"
  Write-Host " Dr. Jinu server + Named Tunnel running"
  Write-Host "========================================"
  Write-Host (" Local    : http://{0}:{1}" -f $BindHost, $Port)
  if (@($LanUrls).Count -gt 0) {
    Write-Host " LAN      :"
    foreach ($url in $LanUrls) {
      Write-Host ("           {0}" -f $url)
    }
  }
  Write-Host (" Public   : {0}" -f $PublicUrl)
  Write-Host (" Health   : {0}" -f $HealthUrl)
  Write-Host (" Admin    : http://{0}:{1}/admin/visitors" -f $BindHost, $Port)
  Write-Host (" Tunnel   : {0}" -f $TunnelName)
  if ($HealthPayload) {
    $serviceStatus = if ($HealthPayload.ready) { "READY" } else { "NOT READY" }
    $defaultProvider = $HealthPayload.llm_provider.default
    $localReady = $HealthPayload.llm_provider.local.ready
    $modelExists = $HealthPayload.model.exists
    $googleReady = $HealthPayload.google_credentials.configured
    Write-Host (" Status   : {0}" -f $serviceStatus)
    Write-Host (" LLM      : default={0}, local_ready={1}, model_exists={2}" -f $defaultProvider, $localReady, $modelExists)
    Write-Host (" STT      : google_credentials={0}" -f $googleReady)
  }
  Write-Host ""
  Write-Host "Commands:"
  Write-Host " - status  : 현재 주소 다시 보기"
  Write-Host " - stop    : 서버와 터널 종료"
  Write-Host " - restart : 서버와 터널 재시작"
  Write-Host ""
  Write-Host "Logs:"
  Write-Host (" - {0}" -f $serverOutLog)
  Write-Host (" - {0}" -f $tunnelOutLog)
  Write-Host (" - {0}" -f $tunnelErrLog)
  Write-Host ""
}

$pythonPath = Resolve-CommandPath -CommandName "python"
if (-not $pythonPath) {
  throw "python command was not found."
}

$cloudflaredPath = Get-CloudflaredPath
$appScriptPath = Join-Path $projectRoot "app.py"

Import-DotEnvFile -Path (Join-Path $projectRoot ".env.local")
Import-DotEnvFile -Path (Join-Path $projectRoot ".env")

$resolvedTunnelName = if ($TunnelName) {
  $TunnelName
} else {
  Get-SettingValue -Keys @("CLOUDFLARE_TUNNEL_NAME", "CLOUDFLARED_TUNNEL_NAME", "TUNNEL_NAME") -Default "aidrgnu-named-tunnel"
}

$resolvedPublicHostname = if ($PublicHostname) {
  $PublicHostname
} else {
  Get-SettingValue -Keys @("CLOUDFLARE_PUBLIC_HOSTNAME", "CLOUDFLARED_PUBLIC_HOSTNAME", "TUNNEL_PUBLIC_HOSTNAME") -Default "aidrgnu.com"
}

$resolvedTokenFile = Get-TunnelTokenFilePath -ProjectRoot $projectRoot

if ($ValidateOnly) {
  Write-Host ("Python             : {0}" -f $pythonPath)
  Write-Host ("cloudflared        : {0}" -f $cloudflaredPath)
  Write-Host ("Project            : {0}" -f $projectRoot)
  Write-Host ("Tunnel Name        : {0}" -f $resolvedTunnelName)
  Write-Host ("Public Hostname    : {0}" -f $resolvedPublicHostname)
  Write-Host ("Tunnel Token File  : {0}" -f $(if ($resolvedTokenFile) { $resolvedTokenFile } else { "NOT FOUND" }))
  exit 0
}

if (-not $resolvedTokenFile) {
  throw ("Named Tunnel 토큰을 찾지 못했습니다.`n`n" +
    "다음 중 하나를 설정해 주세요.`n" +
    "1. .env.local 에 CLOUDFLARE_TUNNEL_TOKEN=... 추가`n" +
    "2. data\\cloudflared_tunnel_token.txt 파일 생성`n" +
    "3. -TunnelTokenFile 인자로 토큰 파일 경로 전달")
}

$jobHandle = $null
$serverProcess = $null
$tunnelProcess = $null
$shouldRestart = $false

Stop-ExistingManagedProcesses -Port $Port -TunnelName $resolvedTunnelName -TunnelLogPath $tunnelOutLog -TokenFilePath $resolvedTokenFile

try {
  $jobHandle = New-KillOnCloseJobHandle

  Remove-Item $serverOutLog, $serverErrLog, $tunnelOutLog, $tunnelStdOutLog, $tunnelErrLog -Force -ErrorAction SilentlyContinue

  $serverProcess = Start-Process `
    -FilePath $pythonPath `
    -ArgumentList $appScriptPath `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $serverOutLog `
    -RedirectStandardError $serverErrLog `
    -PassThru `
    -WindowStyle Hidden

  [System.IO.File]::WriteAllText(
    $serverPidFile,
    [string]$serverProcess.Id,
    [System.Text.UTF8Encoding]::new($false)
  )
  Add-ProcessToJob -JobHandle $jobHandle -Process $serverProcess

  $healthUrl = "http://{0}:{1}/health" -f $BindHost, $Port

  Write-Host ""
  Write-Step "1/5 기존 서버 및 터널 프로세스를 정리했습니다."
  Write-Step "2/5 Dr. Jinu 로컬 서버를 시작합니다."
  Write-Host ("      server stdout log: {0}" -f $serverOutLog)
  Write-Host ("      server stderr log: {0}" -f $serverErrLog)

  [void](Wait-ForHealth -Url $healthUrl)
  Write-Step "3/5 로컬 서버 헬스체크가 완료되었습니다."

  $healthPayload = Get-HealthPayload -Url $healthUrl
  $lanUrls = Get-LanUrls -Port $Port

  $tunnelArgs = @(
    "tunnel",
    "--no-autoupdate",
    "--loglevel", "info",
    "--logfile", $tunnelOutLog,
    "run",
    "--token-file", $resolvedTokenFile
  )

  $tunnelProcess = Start-Process `
    -FilePath $cloudflaredPath `
    -ArgumentList $tunnelArgs `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $tunnelStdOutLog `
    -RedirectStandardError $tunnelErrLog `
    -PassThru `
    -WindowStyle Hidden

  [System.IO.File]::WriteAllText(
    $tunnelPidFile,
    [string]$tunnelProcess.Id,
    [System.Text.UTF8Encoding]::new($false)
  )
  Add-ProcessToJob -JobHandle $jobHandle -Process $tunnelProcess

  Write-Step "4/5 Cloudflare Named Tunnel을 여는 중입니다."
  Write-Host ("      cloudflared log   : {0}" -f $tunnelOutLog)
  Write-Host ("      cloudflared stdout: {0}" -f $tunnelStdOutLog)
  Write-Host ("      cloudflared stderr: {0}" -f $tunnelErrLog)

  [void](Wait-ForNamedTunnelReady -LogPaths @($tunnelOutLog, $tunnelStdOutLog, $tunnelErrLog) -Process $tunnelProcess)
  Write-Step "5/5 공개 도메인에 연결되는 Named Tunnel이 준비되었습니다."

  $publicUrl = "https://{0}" -f $resolvedPublicHostname
  Write-StatusBlock -BindHost $BindHost -Port $Port -LanUrls $lanUrls -PublicUrl $publicUrl -HealthUrl $healthUrl -HealthPayload $healthPayload -TunnelName $resolvedTunnelName

  while ($true) {
    if ($serverProcess.HasExited) {
      throw "The app server exited. Check the logs."
    }
    if ($tunnelProcess.HasExited) {
      throw "The Cloudflare Named Tunnel exited. Check the logs."
    }

    $command = Read-Host "command"
    $normalizedCommand = ($command | Out-String).Trim().ToLowerInvariant()

    if ($normalizedCommand -eq "stop") {
      Write-Step "종료 명령을 받았습니다. 서버와 터널을 정리합니다."
      break
    }

    if ($normalizedCommand -eq "restart") {
      Write-Step "재시작 명령을 받았습니다. 서버와 터널을 다시 시작합니다."
      $shouldRestart = $true
      break
    }

    if ($normalizedCommand -eq "status" -or [string]::IsNullOrWhiteSpace($normalizedCommand)) {
      Write-StatusBlock -BindHost $BindHost -Port $Port -LanUrls $lanUrls -PublicUrl $publicUrl -HealthUrl $healthUrl -HealthPayload $healthPayload -TunnelName $resolvedTunnelName
      continue
    }

    Write-Host ("Unknown command: {0}" -f $command)
    Write-Host "Type 'status', 'restart', or 'stop'."
  }
}
finally {
  if ($tunnelProcess -and -not $tunnelProcess.HasExited) {
    try { Stop-Process -Id $tunnelProcess.Id -Force -ErrorAction SilentlyContinue } catch {}
  }

  if ($serverProcess -and -not $serverProcess.HasExited) {
    try { Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue } catch {}
  }

  Remove-Item $serverPidFile, $tunnelPidFile -Force -ErrorAction SilentlyContinue
  if (Test-Path $generatedTokenFile) {
    Remove-Item $generatedTokenFile -Force -ErrorAction SilentlyContinue
  }

  if ($jobHandle -and $jobHandle -ne [IntPtr]::Zero) {
    [KillOnCloseJob.Native]::CloseHandle($jobHandle) | Out-Null
  }
}

if ($shouldRestart) {
  Start-Process -FilePath (Join-Path $projectRoot "start_server_with_tunnel.bat") -WorkingDirectory $projectRoot | Out-Null
}
