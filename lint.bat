@echo off
setlocal

set PATH=%ProgramFiles%\nodejs;%PATH%
npm run lint
if errorlevel 1 exit /b %errorlevel%

echo Lint checks passed.
