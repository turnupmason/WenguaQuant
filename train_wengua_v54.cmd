@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "WENGUA_PY=%~dp0..\WenguaApp\local\qwen_env\Scripts\python.exe"
if not exist "%WENGUA_PY%" (
  echo Missing the existing WenguaApp qwen_env. Keep both repositories side by side.
  pause
  exit /b 2
)
if not exist "local\course_v53_data\pack.json" (
  "%WENGUA_PY%" -X utf8 -m wengua_course_v53 restore
  if errorlevel 1 exit /b 2
)
if not exist "local\course_v54_data\course_delta.json" (
  "%WENGUA_PY%" -X utf8 -m wengua_course_v54 restore
  if errorlevel 1 exit /b 2
)
"%WENGUA_PY%" -X utf8 -m wengua_course_v54 run --resume %*
set "WENGUA_EXIT=%ERRORLEVEL%"
echo Report: local\course_v54\run1\results.html
if "%WENGUA_EXIT%"=="0" (
  echo Planned groups finished. Review all epoch answers and the accepted-candidate status.
) else (
  echo Paused or failed. Outputs and complete checkpoints are preserved. Check the report before resuming.
)
pause
exit /b %WENGUA_EXIT%
