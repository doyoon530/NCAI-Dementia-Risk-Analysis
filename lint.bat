@echo off
setlocal

python -m ruff check app.py ncai_app
if errorlevel 1 exit /b %errorlevel%

python -m compileall -q app.py ncai_app
if errorlevel 1 exit /b %errorlevel%

"%ProgramFiles%\nodejs\node.exe" "%~dp0node_modules\prettier\bin\prettier.cjs" --check "templates/**/*.html" "static/**/*.css" "static/**/*.js"
if errorlevel 1 exit /b %errorlevel%

echo Lint checks passed.
