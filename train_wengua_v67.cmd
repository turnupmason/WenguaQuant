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
  "%WENGUA_PY%" -X utf8 -m wengua_course_v67 --help
  exit /b
)
"%WENGUA_PY%" -X utf8 -m wengua_course_v67 run --resume %*
set "WENGUA_EXIT=%ERRORLEVEL%"
"%WENGUA_PY%" -X utf8 -m wengua_course_v67 export %*
if "%WENGUA_EXIT%"=="0" set "WENGUA_EXIT=%ERRORLEVEL%"
:finish
echo Report: local\course_v67\run1\results.html
echo Original weights and intact checkpoints are preserved. Repeat this command to resume.
pause
exit /b %WENGUA_EXIT%
