@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "WENGUA_PY=%~dp0..\WenguaApp\local\qwen_env\Scripts\python.exe"
if not exist "%WENGUA_PY%" (
  echo Missing the existing WenguaApp qwen_env. Keep the two repositories side by side.
  pause
  exit /b 2
)
"%WENGUA_PY%" -X utf8 -c "from zoneinfo import ZoneInfo; ZoneInfo('Asia/Shanghai')"
if errorlevel 1 (
  "%WENGUA_PY%" -m pip install tzdata==2025.2
  if errorlevel 1 exit /b 2
)
if not exist "local\course_v53_data\pack.json" (
  "%WENGUA_PY%" -X utf8 -m wengua_course_v53 restore
  if errorlevel 1 (
    echo Restore failed. Preserve the error.
    pause
    exit /b 2
  )
)
"%WENGUA_PY%" -X utf8 -m wengua_course_v53 run --resume %*
set "WENGUA_EXIT=%ERRORLEVEL%"
echo Report: local\course_v53\run1\results.html
if not "%WENGUA_EXIT%"=="0" echo Paused or stopped. Preserve the report; repeat to resume the same checkpoint.
pause
exit /b %WENGUA_EXIT%
