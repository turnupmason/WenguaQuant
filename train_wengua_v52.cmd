@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "WENGUA_PY=%~dp0..\WenguaApp\local\qwen_env\Scripts\python.exe"
if not exist "%WENGUA_PY%" (
  echo Missing the existing WenguaApp qwen_env. Place both repositories side by side.
  pause
  exit /b 2
)
if not exist "local\course_v52_data\pack.json" (
  "%WENGUA_PY%" -X utf8 -m wengua_course restore
  if errorlevel 1 (
    echo Course restore failed. Preserve this message for review.
    pause
    exit /b 2
  )
)
"%WENGUA_PY%" -X utf8 -m wengua_course run --resume %*
set "WENGUA_EXIT=%ERRORLEVEL%"
echo Report: local\course_v52\run1\results.html
if not "%WENGUA_EXIT%"=="0" echo Stopped or paused. Outputs preserved; repeat this command to resume.
pause
exit /b %WENGUA_EXIT%
