@echo off
rem 116-C5 -- doble clic o desde cmd: lanzar.bat [-Prueba] [-Memoria 12g]   (llama a lanzar.ps1 sin cambiar la politica del sistema)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0lanzar.ps1" %*
set RC=%ERRORLEVEL%
pause
exit /b %RC%
