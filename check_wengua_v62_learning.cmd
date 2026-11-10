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
  "%WENGUA_PY%" -X utf8 -m research_audits.v62_probe --help
  exit /b
)
"%WENGUA_PY%" -X utf8 -m research_audits.v62_probe run --resume %*
set "WENGUA_EXIT=%ERRORLEVEL%"
if not "%WENGUA_EXIT%"=="0" goto finish
"%WENGUA_PY%" -X utf8 -m research_audits.v62_probe export %*
set "WENGUA_EXIT=%ERRORLEVEL%"
:finish
echo Inference only. No training or weight updates. Results: local\v62_learning_probe\run1\results.html
pause
exit /b %WENGUA_EXIT%
