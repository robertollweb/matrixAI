# 128-C5 -- UNA orden: mide TabM en la CPU (brazo A) y en la GPU (brazo B) en ESTE PC, con el paquete GPU, y deja
# salida\resultado_128_c5.json para mandarlo. Windows con Docker Desktop (WSL2 + GPU NVIDIA). Ver LEEME.md.
# (ASCII a proposito: PowerShell 5.1 lee los .ps1 sin BOM como ANSI.)
#
#   .\lanzar.ps1 -Imagen <imagen del paquete GPU con el corte C5> [-Prueba] [-Memoria 6g] [-Cpus 2]
#                [-Brazos AB] [-Solo a,b] [-Suelo X] [-Salida DIR] [-ImagenMedidor IMG]
#
# Se puede cortar (Ctrl+C) y volver a lanzar con la MISMA orden: sigue donde estaba.
param(
  [string]$Imagen = "",
  [switch]$Prueba,
  [string]$Memoria = "6g",
  [string]$Cpus = "2",
  [string]$Brazos = "AB",
  [string]$Solo = "",
  [string]$Suelo = "",
  [string]$Salida = "",
  [string]$ImagenMedidor = ""
)
$ErrorActionPreference = "Continue"
$Aqui = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($Salida -eq "") { $Salida = Join-Path $Aqui "salida" }
$Brazos = ($Brazos -replace ",", "").ToUpper()
function Log($m) { Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m) }
function Parar($m, $codigo) { Write-Host $m -ForegroundColor Red; Limpiar; exit $codigo }

if ($Imagen -eq "") { Write-Host "Falta -Imagen <imagen del paquete GPU>. Mira 'docker images' (suele llamarse matrixai-studio-gpu...)"; exit 2 }
if ($ImagenMedidor -eq "") { $ImagenMedidor = $Imagen }
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Write-Host "No encuentro docker."; exit 2 }
docker image inspect $Imagen *> $null
if ($LASTEXITCODE -ne 0) { Write-Host "No existe la imagen $Imagen en este Docker (docker images)."; exit 2 }

New-Item -ItemType Directory -Force -Path (Join-Path $Salida "maquina") | Out-Null
$Salida = (Resolve-Path $Salida).Path
$Kit = $Aqui
$Red = "c5-128-red"

function Limpiar {
  docker rm -f c5-128-studio-A c5-128-studio-B c5-128-medidor *> $null
  docker network rm $Red *> $null
}
Limpiar
$ImagenId = (docker image inspect $Imagen --format "{{.Id}}")

# --- 0. la GPU de verdad, dentro de Docker (si no, no se empieza). Solo si se va a medir el brazo B.
if ($Brazos.Contains("B")) {
  Log "Paso 0: comprobando que Docker ve la GPU..."
  $t = docker run --rm --gpus all --entrypoint python3 $Imagen -c "import torch,sys; ok=torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'GPU:', torch.cuda.get_device_name(0) if ok else None); sys.exit(0 if ok else 1)" 2>&1
  $t | Out-File -Encoding ascii (Join-Path $Salida "maquina\torch_gpu.txt")
  Write-Host $t
  if ($LASTEXITCODE -ne 0) { Parar "PARO: Docker no ve la GPU con esta imagen (revisa que Docker Desktop use WSL2 y tenga la GPU NVIDIA habilitada)." 3 }
  docker run --rm --gpus all --entrypoint nvidia-smi $Imagen -q 2>&1 | Out-File -Encoding ascii (Join-Path $Salida "maquina\nvidia-smi.txt")
}
(docker version; "---"; docker info) 2>&1 | Out-File -Encoding ascii (Join-Path $Salida "maquina\docker.txt")
(Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors | Format-List | Out-String),
(Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory | Format-List | Out-String),
(Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version | Format-List | Out-String) |
  Out-File -Encoding ascii (Join-Path $Salida "maquina\sistema.txt")
"$Imagen $ImagenId" | Out-File -Encoding ascii (Join-Path $Salida "maquina\imagen.txt")
docker network create $Red *> $null

# --- 1. verificar los datos del kit (contra el manifiesto, byte a byte)
docker run --rm --entrypoint python3 -v "${Kit}:/work/kit:ro" $ImagenMedidor /work/kit/medir_c5.py --verificar-datos --datos /work/kit/datos
if ($LASTEXITCODE -ne 0) { Parar "PARO: los datos del kit no son los del manifiesto." 2 }

function Medidor($brazo) {
  if ($brazo -eq "A") { $ent = '{"MATRIXAI_TABM_DISPOSITIVO":"cpu"}' } else { $ent = '{"MATRIXAI_TABM_DISPOSITIVO":"auto","docker_gpus":"all"}' }
  # El JSON de --entorno lleva comillas: se pasa por una variable de entorno y no por la linea de ordenes (PowerShell 5.1
  # se come las comillas dobles al llamar a programas externos).
  $argumentos = @("run", "--rm", "--name", "c5-128-medidor", "--network", $Red, "-e", "C5_ENTORNO=$ent", "-e", "HOME=/tmp", "--entrypoint", "python3",
    "-v", "${Kit}:/work/kit:ro", "-v", "${Salida}:/salida", $ImagenMedidor,
    "/work/kit/medir_c5.py", "--brazo", $brazo, "--studio", "http://c5-128-studio-${brazo}:8765", "--salida", "/salida", "--datos", "/work/kit/datos",
    "--imagen", $Imagen, "--imagen-id", $ImagenId, "--memoria", $Memoria, "--cpus", $Cpus, "--entorno-de-env", "C5_ENTORNO")
  if ($Solo -ne "") { $argumentos += @("--solo", $Solo) }
  if ($Suelo -ne "") { $argumentos += @("--suelo", $Suelo) }
  if ($Prueba) { $argumentos += "--prueba" }
  & docker @argumentos
  return $LASTEXITCODE
}

function Brazo($b) {
  for ($intento = 1; $intento -le 3; $intento++) {
    docker rm -f "c5-128-studio-$b" *> $null
    if ($b -eq "B") { $gpu = @("--gpus", "all"); $disp = "auto" } else { $gpu = @(); $disp = "cpu" }
    Log "Brazo ${b}: arrancando el Studio (TabM en '$disp', memoria $Memoria, $Cpus CPU) ..."
    $run = @("run", "-d", "--init", "--name", "c5-128-studio-$b", "--network", $Red, "--memory=$Memoria", "--memory-swap=$Memoria", "--cpus=$Cpus",
      "-e", "MATRIXAI_LICENSE_ENABLED=false", "-e", "MATRIXAI_TABM_DISPOSITIVO=$disp") + $gpu + @("-v", "${Kit}:/work/kit:ro", "-v", "${Salida}:/salida", $Imagen)
    & docker @run *> $null
    if ($LASTEXITCODE -ne 0) { Write-Host "PARO: no arranca el Studio del brazo $b"; return 4 }
    docker exec -d "c5-128-studio-$b" python3 /work/kit/muestreador.py "/salida/muestras_$b.jsonl" *> $null
    $rc = Medidor $b
    docker exec "c5-128-studio-$b" python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'disponible',torch.cuda.is_available())" 2>&1 |
      Out-File -Encoding ascii (Join-Path $Salida "maquina\torch_$b.txt")
    docker logs "c5-128-studio-$b" 2>&1 | Out-File -Encoding ascii (Join-Path $Salida "studio_$b.log")
    docker rm -f "c5-128-studio-$b" *> $null
    if ($rc -eq 4) { Log "El Studio del brazo $b dejo de contestar: lo rearranco y sigo (intento $intento de 3)."; continue }
    return $rc
  }
  return 4
}

if ($Prueba) { Log "PRUEBA: un conjunto pequeno por brazo (unos minutos, estimado). Debe decir 'la GPU SE USO'." }
foreach ($b in @("A", "B")) {
  if ($Brazos.Contains($b)) {
    $rc = Brazo $b
    if ($rc -ne 0) { Parar "El brazo $b termino con codigo $rc." $rc }
  }
}
if ($Prueba) { Log "Prueba hecha. Ahora lanza la medicion de verdad (sin -Prueba)."; Limpiar; exit 0 }
if ($Brazos -ne "AB") { Log "Hecho el brazo $Brazos (falta el otro para componer el resultado)."; Limpiar; exit 0 }
$comp = @("run", "--rm", "-e", "HOME=/tmp", "--entrypoint", "python3", "-v", "${Kit}:/work/kit:ro", "-v", "${Salida}:/salida", $ImagenMedidor,
  "/work/kit/medir_c5.py", "--componer", "--salida", "/salida", "--datos", "/work/kit/datos")
if ($Suelo -ne "") { $comp += @("--suelo", $Suelo) }
& docker @comp
Log "Hecho. Manda: $Salida\resultado_128_c5.json"
Limpiar
