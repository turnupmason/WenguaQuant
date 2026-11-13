@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "PYTHONDONTWRITEBYTECODE=1"
if not defined WENGUA_PY (
 set "WENGUA_PY=%USERPROFILE%\WenguaApp\local\qwen_env\Scripts\python.exe"
 if defined WENGUA_APP_ROOT set "WENGUA_PY=%WENGUA_APP_ROOT%\local\qwen_env\Scripts\python.exe"
)
if not exist "%WENGUA_PY%" (
 echo Cannot find the existing qwen_env. Set WENGUA_APP_ROOT to your WenguaApp folder.
 pause
 exit /b 1
)
if "%~1"=="--help" (
 "%WENGUA_PY%" -X utf8 -m wengua_assistant.server --help
 exit /b
)
"%WENGUA_PY%" -X utf8 -m wengua_assistant.check
if errorlevel 1 goto failed
"%WENGUA_PY%" -X utf8 -m wengua_assistant.server %*
if errorlevel 1 goto failed
exit /b 0
:failed
echo Stopped. Existing weights, training runs and private sources are preserved.
pause
exit /b 1
