@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
if not defined WENGUA_PY set "WENGUA_PY=%~dp0..\WenguaApp\local\qwen_env\Scripts\python.exe"
if not exist "%WENGUA_PY%" (
  echo Missing WenguaApp local qwen_env. Keep both repositories side by side.
  pause
  exit /b 2
)
if "%~1"=="--help" (
  "%WENGUA_PY%" -X utf8 -m wengua_recovery.v59_watchdog --help
  exit /b
)
"%WENGUA_PY%" -X utf8 -m wengua_recovery.v59_watchdog check
if errorlevel 1 exit /b 2
"%WENGUA_PY%" -X utf8 -m wengua_recovery.v59_watchdog run %*
set "WENGUA_EXIT=%ERRORLEVEL%"
echo Report: local\course_v59\run1\results.html
echo Original weights and intact checkpoints are preserved. Repeat this command to resume.
pause
exit /b %WENGUA_EXIT%
