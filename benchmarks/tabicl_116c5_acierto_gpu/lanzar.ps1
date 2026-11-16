# 116-C5 -- UNA orden: construye la imagen (la 1.a vez), comprueba la GPU y mide el acierto de TabICL en ESTE PC.
# Windows con Docker Desktop (WSL2 + GPU NVIDIA). Deja salida\resultado_116c5_acierto_gpu.json y salida\log_acierto_gpu.txt.
# (ASCII a proposito: PowerShell 5.1 lee los .ps1 sin BOM como ANSI.)
#
#   .\lanzar.ps1              # construye si hace falta, prueba la GPU (1 minuto) y mide TODO
#   .\lanzar.ps1 -Prueba      # solo la prueba: 1 conjunto, 1 pliegue (salida\prueba.json)
#   .\lanzar.ps1 -Memoria 12g # tope de RAM del contenedor (por omision 12g)
#
# Se puede cortar (Ctrl+C) y volver a lanzar con la MISMA orden: sigue donde estaba (los pliegues ya medidos se reusan).
param(
  [switch]$Prueba,
  [string]$Memoria = "12g"
)
$ErrorActionPreference = "Continue"
$aqui = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $aqui
$imagen = "medicion-tabicl-acierto-gpu:2.2.0"
New-Item -ItemType Directory -Force -Path "$aqui\salida" | Out-Null

docker image inspect $imagen *> $null
if ($LASTEXITCODE -ne 0) {
  Write-Host "Construyendo la imagen (la primera vez: ~3 GB de PyTorch con CUDA y 215 MB de pesos, 10-40 min)..."
  docker build -t $imagen .
  if ($LASTEXITCODE -ne 0) { Write-Host "FALLO al construir la imagen (ver arriba)"; exit 1 }
}

Write-Host "Comprobando que Docker ve la GPU..."
docker run --rm --gpus all $imagen python3 -c "import torch,sys; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NINGUNA'); sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) { Write-Host "Docker NO ve la GPU: revisa el driver de NVIDIA y que Docker Desktop use el motor WSL 2 (LEEME.md, paso 1)"; exit 1 }

if ($Prueba) {
  docker run --rm --gpus all --network none --memory=$Memoria -v "${aqui}\salida:/salida" $imagen python3 medir_acierto.py --prueba --salida /salida/prueba.json
  exit $LASTEXITCODE
}
docker run --rm --gpus all --network none --memory=$Memoria -v "${aqui}\salida:/salida" $imagen
$rc = $LASTEXITCODE
Write-Host "Terminado (codigo $rc). Mandame salida\resultado_116c5_acierto_gpu.json y salida\log_acierto_gpu.txt"
exit $rc
