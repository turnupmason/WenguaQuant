@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
if not defined WENGUA_PY set "WENGUA_PY=%~dp0..\WenguaApp\local\qwen_env\Scripts\python.exe"
if not exist "%WENGUA_PY%" (
  echo Missing WenguaApp local qwen_env. Keep WenguaQuant and WenguaApp side by side.
  pause
  exit /b 2
)
if "%~1"=="--help" (
  "%WENGUA_PY%" -X utf8 -m wengua_course_v57 --help
  exit /b
)
if not exist "local\course_v57_data\course_delta.json.gz" (
  "%WENGUA_PY%" -X utf8 -m wengua_course_v57 restore
  if errorlevel 1 exit /b 2
)
"%WENGUA_PY%" -X utf8 -m wengua_course_v57 check
if errorlevel 1 exit /b 2
"%WENGUA_PY%" -X utf8 -m wengua_course_v57 run --resume %*
set "WENGUA_EXIT=%ERRORLEVEL%"
echo Report: local\course_v57\run1\results.html
echo Read run_status.json: needs_course_review means mixed-pilot retention needs review; full training was not started.
echo Outputs and complete checkpoints are preserved. Repeat this command to resume a paused run.
pause
exit /b %WENGUA_EXIT%
