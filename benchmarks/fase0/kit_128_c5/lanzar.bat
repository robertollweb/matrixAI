@echo off
rem 128-C5 -- doble clic o desde cmd: lanzar.bat -Imagen <imagen> [-Prueba] ...   (llama a lanzar.ps1 sin cambiar la politica del sistema)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0lanzar.ps1" %*
pause
