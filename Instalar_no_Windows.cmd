@echo off
setlocal
chcp 65001 >nul
if "%~1"=="" (
    if exist "%~dp0MosquitoWingbeat-Windows.exe" (
        start "" "%~dp0MosquitoWingbeat-Windows.exe"
    ) else (
        echo Baixando o aplicativo grafico: nao precisa instalar Python.
        start "" "https://github.com/Lciarallo/mosquito-wingbeat/releases/latest/download/MosquitoWingbeat-Windows.exe"
    )
    exit /b 0
)
set "TASK_DEFAULT_OPTION="
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
echo Para uso sem comandos, abra este atalho sem argumentos e baixe o aplicativo grafico.
echo Para usar estas opcoes avancadas por terminal, instale Python 3.9 ou mais recente.
pause
exit /b 1
:finished
set "TASK_INSTALL_EXIT=%errorlevel%"
echo.
pause
exit /b %TASK_INSTALL_EXIT%
