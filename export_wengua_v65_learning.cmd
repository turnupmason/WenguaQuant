@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not defined WENGUA_PY set "WENGUA_PY=%~dp0..\WenguaApp\local\qwen_env\Scripts\python.exe"
"%WENGUA_PY%" -X utf8 -m research_audits.v65_learning_probe export %*
pause
