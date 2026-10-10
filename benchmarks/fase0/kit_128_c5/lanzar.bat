@echo off
rem 128-C5 -- doble clic o desde cmd: lanzar.bat [-Prueba] [-Brazos A] ...   (llama a lanzar.ps1 sin cambiar la politica del sistema)
rem Ya no hace falta -Imagen: el kit carga su imagen y construye la capa GPU (ver LEEME.md).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0lanzar.ps1" %*
set RC=%ERRORLEVEL%
pause
exit /b %RC%
