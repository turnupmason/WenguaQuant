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
  "%WENGUA_PY%" -X utf8 -m wengua_course_v60 --help
  exit /b
)
"%WENGUA_PY%" -X utf8 -m wengua_course_v60 check >nul 2>nul
if errorlevel 1 (
  "%WENGUA_PY%" -X utf8 -m wengua_course_v60 restore
  if errorlevel 1 exit /b 2
)
"%WENGUA_PY%" -X utf8 -m wengua_course_v60 run --resume %*
set "WENGUA_EXIT=%ERRORLEVEL%"
echo Report: local\course_v60\run1\results.html
echo Three one-epoch rate trials use the same retained parent. Only passing candidates continue.
echo Weights and optimizer checkpoints are preserved. Repeat this command to resume.
pause
exit /b %WENGUA_EXIT%
