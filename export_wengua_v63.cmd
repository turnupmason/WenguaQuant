@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
if not defined WENGUA_PY set "WENGUA_PY=%~dp0..\WenguaApp\local\qwen_env\Scripts\python.exe"
"%WENGUA_PY%" -X utf8 -m wengua_course_v63 export %*
pause
