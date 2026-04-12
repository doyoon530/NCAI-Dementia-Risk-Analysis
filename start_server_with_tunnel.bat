@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "POWERSHELL_CMD=powershell"
where pwsh >nul 2>nul
if %errorlevel%==0 set "POWERSHELL_CMD=pwsh"

"%POWERSHELL_CMD%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_server_with_tunnel.ps1"
