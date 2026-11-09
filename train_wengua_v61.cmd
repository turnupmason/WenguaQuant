@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
if not defined WENGUA_PY set "WENGUA_PY=%~dp0..\WenguaApp\local\qwen_env\Scripts\python.exe"
if not exist "%WENGUA_PY%" (
  echo Missing sibling WenguaApp local qwen_env.
  pause
  exit /b 2
)
if "%~1"=="--help" (
  "%WENGUA_PY%" -X utf8 -m wengua_course_v61 --help
  exit /b
)
"%WENGUA_PY%" -X utf8 -m wengua_course_v61 check >nul 2>nul
if errorlevel 1 (
  "%WENGUA_PY%" -X utf8 -m wengua_course_v61 restore
  if errorlevel 1 exit /b 2
)
"%WENGUA_PY%" -X utf8 -m wengua_course_v61 run --resume %*
set "WENGUA_EXIT=%ERRORLEVEL%"
echo Report: local\course_v61\run1\results.html
echo Two learning rates, two epochs each. All checkpoints and original weights are preserved.
echo Repeat this command to resume an interrupted run.
pause
exit /b %WENGUA_EXIT%
