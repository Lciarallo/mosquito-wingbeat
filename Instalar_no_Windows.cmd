@echo off
setlocal
chcp 65001 >nul
set "TASK_DEFAULT_OPTION="
if "%~1"=="" set "TASK_DEFAULT_OPTION=--monitor"
where py >nul 2>nul
if not errorlevel 1 (
    py -3 "%~dp0install_arduino.py" %TASK_DEFAULT_OPTION% %*
    goto finished
)
where python >nul 2>nul
if not errorlevel 1 (
    python "%~dp0install_arduino.py" %TASK_DEFAULT_OPTION% %*
    goto finished
)
echo Instale Python 3.9 ou mais recente de https://www.python.org/downloads/
echo ou use a Arduino IDE conforme LEIA_PRIMEIRO.md / o manual.
pause
exit /b 1
:finished
set "TASK_INSTALL_EXIT=%errorlevel%"
echo.
pause
exit /b %TASK_INSTALL_EXIT%
