# 128-C5 -- UNA orden: mide TabM en la CPU (brazo A) y en la GPU (brazo B) en ESTE PC, con el paquete GPU, y deja
# salida\resultado_128_c5.json para mandarlo. Windows con Docker Desktop (WSL2 + GPU NVIDIA). Ver LEEME.md.
# (ASCII a proposito: PowerShell 5.1 lee los .ps1 sin BOM como ANSI.)
# Vale para Windows PowerShell 5.1 (el que abre lanzar.bat) y para PowerShell 7 (pwsh). Lo prueba
# tests/test_128_c5_kit.py con pwsh 7 y un docker falso, tambien con el paso de argumentos "Legacy" de 5.1.
#
#   .\lanzar.ps1 [-Prueba] [-Memoria 6g] [-Cpus 2] [-Brazos AB] [-Solo a,b] [-Suelo X] [-Salida DIR]
#                [-SinCortar] [-Fases | -SoloFases] [-ConjuntosFases a,b] [-PlazoFases 1800]
#                [-Reconstruir] [-Imagen IMG -ImagenMedidor IMG]
#
# SIN -Imagen (lo normal): carga la imagen BASE de este corte (imagen\matrixai-studio-128.tar) y CONSTRUYE en este PC
# la capa GPU (imagen\Dockerfile.gpu: PyTorch con CUDA 12.4, ~3 GB de descarga), igual que el instalador del paquete GPU.
# Con -Imagen se usa una ya hecha y no se construye nada.
# Antes de medir COMPRUEBA que la base de este Docker es la de ESTE kit (su ID contra imagen\matrixai-studio-128.ids) y
# que la capa se construyo sobre ella; si no, carga el tar y rehace la capa solo (128, I3 de la auditoria final: el kit 2
# reusaba en silencio la imagen del kit 1, con las mismas etiquetas).
#
# Los DOS brazos van con --gpus all; solo cambia donde entrena TabM (A: cpu, B: auto). 128, I2 de la auditoria final.
#
# -Fases (OPCIONAL, DESPUES de los dos brazos): mide TabM POR FASES en la GPU (128-R4: carga, preparacion, red, epocas,
# validacion...) en 5 conjuntos y deja salida\tabm_por_fases_gpu.json. -SoloFases: solo ese paso (con los brazos ya hechos).
#
# Se puede cortar (Ctrl+C) y volver a lanzar con la MISMA orden: sigue donde estaba.
param(
  [string]$Imagen = "",
  [switch]$Prueba,
  [switch]$SinCortar,
  [string]$Memoria = "6g",
  [string]$Cpus = "2",
  [string[]]$Brazos = "AB",
  [string[]]$Solo = @(),
  [string]$Suelo = "",
  [string]$Salida = "",
  [string]$ImagenMedidor = "",
  [switch]$Reconstruir,
  [switch]$Fases,
  [switch]$SoloFases,
  [string[]]$ConjuntosFases = @(),
  [int]$PlazoFases = 1800
)
$ErrorActionPreference = "Continue"
# La salida del medidor (python, UTF-8) se lee bien en la consola de Windows.
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$Aqui = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($Salida -eq "") { $Salida = Join-Path $Aqui "salida" }
# N1 (2.a auditoria del 128): en la consola de PowerShell, "-Solo a,b" y "-Brazos A,B" llegan como una LISTA (no como
# "a,b"); por eso son [string[]] y se UNEN aqui. Por lanzar.bat (-File) llegan como un solo texto "a,b": tambien vale.
[string]$Brazos = (($Brazos -join ",") -replace ",", "").ToUpper()
[string]$Solo = (($Solo | Where-Object { $_ -ne "" }) -join ",")
# Lo mismo para -ConjuntosFases (paso opcional): una LISTA en la consola, un solo texto "a,b" por -File / lanzar.bat. Sin nada: los 5 de siempre.
[string]$ConjuntosFases = (($ConjuntosFases | Where-Object { $_ -ne "" }) -join ",")
if ($ConjuntosFases -eq "") { $ConjuntosFases = "APSFailure,KDDCup09_appetency,Allstate_Claims_Severity,house_16H,mfeat-factors" }
if ($SoloFases) { $Fases = $true; $Brazos = "" }
function Log($m) { Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m) }
function Parar($m, $codigo) { Write-Host $m -ForegroundColor Red; Limpiar; exit $codigo }

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Write-Host "No encuentro docker."; exit 2 }
$Base = "matrixai-studio:128"            # la etiqueta que lleva el tar de imagen\ (build-image.sh --image matrixai-studio:128)
$Capa = "matrixai-studio-gpu-128:local"  # la imagen GPU que se construye AQUI; no pisa ninguna de tu instalacion normal

New-Item -ItemType Directory -Force -Path (Join-Path $Salida "maquina") | Out-Null
$Salida = (Resolve-Path $Salida).Path
$Kit = $Aqui
$Red = "c5-128-red"

function Limpiar {
  docker rm -f c5-128-studio-A c5-128-studio-B c5-128-medidor c5-128-fases *> $null
  docker network rm $Red *> $null
}
Limpiar

# --- -1. la imagen: base (tar) + capa GPU construida en ESTE PC (salvo -Imagen)
# 128, I3 de la auditoria final: NO basta con que las etiquetas existan (las del kit anterior son las mismas). La base
# tiene que ser la del tar de ESTE kit (su ID en imagen\matrixai-studio-128.ids, que apunta el empaquetador leyendo el
# tar) y la capa, construida sobre ESA base (la etiqueta $EtiquetaDeLaBase que se le pone al construirla). Los formatos
# de docker van SIN comillas dobles dentro: PowerShell 5.1 se las come al pasarlos a un programa externo (B2).
$Ids = Join-Path $Aqui "imagen\matrixai-studio-128.ids"
$Tar = Join-Path $Aqui "imagen\matrixai-studio-128.tar"
$EtiquetaDeLaBase = "matrixai.kit.base"
# 128, m7 de la re-auditoria final: la etiqueta de la base NO cubria el Dockerfile.gpu: un kit con la misma base y otro
# Dockerfile.gpu reusaba la capa vieja. Segunda etiqueta con el sha256 del Dockerfile.gpu del kit; la capa vale si
# lleva LAS DOS (si cambia el Dockerfile, se rehace). Una capa de antes (sin esta etiqueta) tambien se rehace.
$EtiquetaDelDockerfile = "matrixai.kit.dockerfile"
function IdDe($img) {
  $id = docker image inspect $img --format "{{.Id}}" 2>$null
  if ($LASTEXITCODE -ne 0) { return "" }
  return ("" + $id).Trim()
}
function EsLaBaseDelKit {
  $id = IdDe $Base
  if ($id -eq "") { return $false }
  return ((Get-Content $Ids | ForEach-Object { $_.Trim() }) -contains $id)
}
function CapaSobre($baseId, $dockerfileSha) {
  $etiquetas = docker image inspect $Capa --format "{{json .Config.Labels}}" 2>$null
  if ($LASTEXITCODE -ne 0) { return $false }
  $texto = "" + $etiquetas
  return ($texto.Contains('"' + $EtiquetaDeLaBase + '":"' + $baseId + '"') -and
          $texto.Contains('"' + $EtiquetaDelDockerfile + '":"' + $dockerfileSha + '"'))
}
if ($Imagen -eq "") {
  if ($Reconstruir) { docker rmi -f $Capa $Base *> $null }
  if (-not (Test-Path $Ids)) { Parar "PARO: falta $Ids (la identidad de la imagen base, que el empaquetador apunta junto a imagen\matrixai-studio-128.tar): sin ella no se sabe si la imagen de este Docker es la de este kit. Pide el kit entero o pasa -Imagen." 2 }
  if (-not (EsLaBaseDelKit)) {
    $otra = IdDe $Base
    if ($otra -ne "") { Log "La imagen $Base de este Docker NO es la de este kit ($otra): cargo la del kit y rehago la capa GPU." }
    if (-not (Test-Path $Tar)) { Parar "PARO: falta $Tar (la imagen base de este corte). Pidela o pasa -Imagen." 2 }
    Log "Cargando la imagen base (docker load, ~190 MB)..."
    docker load -i $Tar
    if ($LASTEXITCODE -ne 0) { Parar "PARO: docker load fallo." 2 }
    docker image inspect $Base *> $null
    if ($LASTEXITCODE -ne 0) { Parar "PARO: el tar no trae la etiqueta $Base." 2 }
    if (-not (EsLaBaseDelKit)) { Parar ("PARO: cargado el tar, $Base es " + (IdDe $Base) + " y el kit apunta otra ($Ids): el tar no es el de este kit.") 2 }
  }
  $BaseId = IdDe $Base
  $dockerfile = Join-Path $Aqui "imagen\Dockerfile.gpu"
  if (-not (Test-Path $dockerfile)) { Parar "PARO: falta $dockerfile." 2 }
  $DockerfileSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $dockerfile).Hash.ToLower()
  if (-not (CapaSobre $BaseId $DockerfileSha)) {
    docker image inspect $Capa *> $null
    if ($LASTEXITCODE -eq 0) { Log "La capa GPU $Capa de este Docker no se construyo sobre la base de este kit con su Dockerfile.gpu: la rehago." }
    $capaDir = Join-Path $Salida "maquina\capa"
    New-Item -ItemType Directory -Force -Path $capaDir | Out-Null
    $texto = [System.IO.File]::ReadAllText($dockerfile, [System.Text.Encoding]::UTF8) -replace "(?m)^FROM .*$", "FROM $Base"
    [System.IO.File]::WriteAllText((Join-Path $capaDir "Dockerfile"), $texto, (New-Object System.Text.UTF8Encoding($false)))
    Log "Construyendo la capa GPU (descarga PyTorch con CUDA, ~3 GB: tarda; se hace UNA vez por imagen base)..."
    docker build -t $Capa --label "$EtiquetaDeLaBase=$BaseId" --label "$EtiquetaDelDockerfile=$DockerfileSha" $capaDir 2>&1 | Tee-Object -FilePath (Join-Path $Salida "maquina\build_capa.log")
    if (-not (CapaSobre $BaseId $DockerfileSha)) { Parar "PARO: no se pudo construir la capa GPU sobre la base y el Dockerfile.gpu de este kit (mira $Salida\maquina\build_capa.log; suele ser la red o el disco)." 2 }
  }
  $Imagen = $Capa
}
if ($ImagenMedidor -eq "") { $ImagenMedidor = $Imagen }
docker image inspect $Imagen *> $null
if ($LASTEXITCODE -ne 0) { Parar "No existe la imagen $Imagen en este Docker (docker images)." 2 }
$ImagenId = (docker image inspect $Imagen --format "{{.Id}}")

# --- 0. la GPU de verdad, dentro de Docker (si no, no se empieza). Para cualquier brazo: los DOS llevan --gpus all.
if ($Brazos -ne "" -or $Fases) {
  Log "Paso 0: comprobando que Docker ve la GPU..."
  $t = docker run --rm --gpus all --entrypoint python3 $Imagen -c "import torch,sys; ok=torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'GPU:', torch.cuda.get_device_name(0) if ok else None); sys.exit(0 if ok else 1)" 2>&1
  $t | Out-File -Encoding ascii (Join-Path $Salida "maquina\torch_gpu.txt")
  Write-Host $t
  if ($LASTEXITCODE -ne 0) { Parar "PARO: Docker no ve la GPU con esta imagen (revisa que Docker Desktop use WSL2 y tenga la GPU NVIDIA habilitada)." 3 }
  docker run --rm --gpus all --entrypoint nvidia-smi $Imagen -q 2>&1 | Out-File -Encoding ascii (Join-Path $Salida "maquina\nvidia-smi.txt")
}
# Un bloque { } y no ( ): dentro de ( ) PowerShell no admite varias ordenes con ";" y el guion entero no compilaba
# (auditoria del 128, B2: ParseFile daba 2 errores en esta linea y lanzar.ps1 moria antes de hacer nada).
& { docker version; "---"; docker info } 2>&1 | Out-File -Encoding ascii (Join-Path $Salida "maquina\docker.txt")
if (Get-Command Get-CimInstance -ErrorAction SilentlyContinue) {
  (Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors | Format-List | Out-String),
  (Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory | Format-List | Out-String),
  (Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version | Format-List | Out-String) |
    Out-File -Encoding ascii (Join-Path $Salida "maquina\sistema.txt")
} else {
  "sin Get-CimInstance (no es Windows)" | Out-File -Encoding ascii (Join-Path $Salida "maquina\sistema.txt")
}
"$Imagen $ImagenId" | Out-File -Encoding ascii (Join-Path $Salida "maquina\imagen.txt")
docker network create $Red *> $null

# --- 1. verificar los datos del kit (contra el manifiesto, byte a byte)
docker run --rm --entrypoint python3 -v "${Kit}:/work/kit:ro" $ImagenMedidor /work/kit/medir_c5.py --verificar-datos --datos /work/kit/datos
if ($LASTEXITCODE -ne 0) { Parar "PARO: los datos del kit no son los del manifiesto." 2 }

function Medidor($brazo) {
  # los dos con la GPU a la vista (I2): el entorno que apunta el medidor lo dice en los dos
  if ($brazo -eq "A") { $ent = '{"MATRIXAI_TABM_DISPOSITIVO":"cpu","gpus":"all"}' } else { $ent = '{"MATRIXAI_TABM_DISPOSITIVO":"auto","gpus":"all"}' }
  # El JSON lleva comillas dobles y PowerShell 5.1 se las come al pasar argumentos a un programa externo: llegaba
  # C5_ENTORNO={MATRIXAI_TABM_DISPOSITIVO:cpu} y el medidor no lo podia leer (B2). Por eso NO va en la linea de
  # ordenes: va en el entorno de ESTE proceso y "-e C5_ENTORNO" (sin "=") le dice a docker que lo copie de ahi.
  $env:C5_ENTORNO = $ent
  $argumentos = @("run", "--rm", "--name", "c5-128-medidor", "--network", $Red, "-e", "C5_ENTORNO", "-e", "HOME=/tmp", "--entrypoint", "python3",
    "-v", "${Kit}:/work/kit:ro", "-v", "${Salida}:/salida", $ImagenMedidor,
    "/work/kit/medir_c5.py", "--brazo", $brazo, "--studio", "http://c5-128-studio-${brazo}:8765", "--salida", "/salida", "--datos", "/work/kit/datos",
    "--imagen", $Imagen, "--imagen-id", $ImagenId, "--memoria", $Memoria, "--cpus", $Cpus, "--entorno-de-env", "C5_ENTORNO")
  if ($Solo -ne "") { $argumentos += @("--solo", $Solo) }
  if ($Suelo -ne "") { $argumentos += @("--suelo", $Suelo) }
  if ($Prueba) { $argumentos += "--prueba" }
  if ($SinCortar) { $argumentos += "--sin-cortar" }
  # "| Out-Host": lo que escribe el medidor va a la PANTALLA (el progreso de horas). Sin eso, dentro de una funcion
  # es su valor de retorno: "$rc = Medidor $b" se lo tragaba entero, $rc era una lista, "-ne 0" salia cierto y el
  # kit paraba tras el brazo A sin haber arrancado el B (B2). La funcion devuelve SOLO el codigo de salida.
  & docker @argumentos | Out-Host
  $codigo = [int]$LASTEXITCODE
  Remove-Item Env:C5_ENTORNO -ErrorAction SilentlyContinue
  return $codigo
}

function Brazo($b) {
  for ($intento = 1; $intento -le 3; $intento++) {
    docker rm -f "c5-128-studio-$b" *> $null
    # 128, I2 de la auditoria final: los DOS brazos ven la GPU (--gpus all) y solo cambia donde entrena TabM. Asi la
    # comparacion aisla a TabM: con la GPU a la vista y TabM en cpu, la densa la esconde como en la v2.12 (A); con
    # TabM en auto no la esconde y el coste de verla va con TabM en la GPU (B).
    $gpu = @("--gpus", "all")
    if ($b -eq "B") { $disp = "auto" } else { $disp = "cpu" }
    Log "Brazo ${b}: arrancando el Studio (TabM en '$disp', memoria $Memoria, $Cpus CPU) ..."
    $run = @("run", "-d", "--init", "--name", "c5-128-studio-$b", "--network", $Red, "--memory=$Memoria", "--memory-swap=$Memoria", "--cpus=$Cpus",
      "-e", "MATRIXAI_LICENSE_ENABLED=false", "-e", "MATRIXAI_TABM_DISPOSITIVO=$disp") + $gpu + @("-v", "${Kit}:/work/kit:ro", "-v", "${Salida}:/salida", $Imagen)
    & docker @run *> $null
    if ($LASTEXITCODE -ne 0) { Write-Host "PARO: no arranca el Studio del brazo $b"; return 4 }
    docker exec -d "c5-128-studio-$b" python3 /work/kit/muestreador.py "/salida/muestras_$b.jsonl" *> $null
    $rc = [int](Medidor $b | Select-Object -Last 1)
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
    $rc = [int](Brazo $b | Select-Object -Last 1)
    if ($rc -ne 0) { Parar "El brazo $b termino con codigo $rc." $rc }
  }
}
# --- 3 (OPCIONAL, -Fases): TabM POR FASES en la GPU. La sonda del 128-R4 corre DENTRO de la imagen GPU (no hace falta Python en
# este PC) con --gpus all y el mismo techo de memoria; los CSV son los del kit (sha256 comprobado). Deja
# salida\tabm_por_fases_gpu.json. Va DESPUES de componer: si falla, el resultado principal ya esta escrito.
# Todo argumento va en un ARRAY (& docker @args): lo que lleva comas ("a,b") es un solo texto, sin comillas dobles que 5.1 pierda.
function PasoFases {
  $conj = $ConjuntosFases; $plazo = $PlazoFases
  if ($Prueba) { $conj = "dresses-sales"; $plazo = 120 }
  Log "Paso opcional: TabM por fases en la GPU ($conj; plazo amplio $plazo s; memoria $Memoria)..."
  docker rm -f c5-128-fases *> $null
  $args_ = @("run", "--rm", "--init", "--name", "c5-128-fases", "--gpus", "all", "--memory=$Memoria", "--memory-swap=$Memoria", "--cpus=$Cpus",
    "-e", "HOME=/tmp", "-e", "MATRIXAI_LICENSE_ENABLED=false", "--entrypoint", "python3",
    "-v", "${Kit}:/work/kit:ro", "-v", "${Salida}:/salida", $Imagen,
    "/work/kit/sonda_128_c0.py", "--tabm-por-fases", "--gpu", "--dentro-del-contenedor", "--datos-del-kit", "/work/kit/datos",
    "--temporales", "/salida/fases_tmp", "--salida", "/salida/tabm_por_fases_gpu.json", "--imagen", $Imagen, "--imagen-id", $ImagenId,
    "--memoria", $Memoria, "--plazo-amplio", "$plazo", "--tope-estudio", "3600", "--solo", $conj)
  & docker @args_ | Out-Host
  $codigo = [int]$LASTEXITCODE
  if ($codigo -eq 0) { Log "Hecho. Manda tambien: $Salida\tabm_por_fases_gpu.json (y la carpeta tabm_por_fases_gpu_series\)." }
  else { Write-Host "El paso de fases termino con codigo $codigo (lo ya medido esta en $Salida\tabm_por_fases_gpu.json)." }
  return $codigo
}

$componer = $true
if ($SoloFases) { $componer = $false }
elseif ($Prueba) { Log "Prueba hecha. Ahora lanza la medicion de verdad (sin -Prueba)."; $componer = $false }
elseif ($Brazos -ne "AB") { Log "Hecho el brazo $Brazos (falta el otro para componer el resultado)."; $componer = $false; $Fases = $false }
if ($componer) {
  $comp = @("run", "--rm", "-e", "HOME=/tmp", "--entrypoint", "python3", "-v", "${Kit}:/work/kit:ro", "-v", "${Salida}:/salida", $ImagenMedidor,
    "/work/kit/medir_c5.py", "--componer", "--salida", "/salida", "--datos", "/work/kit/datos")
  if ($Suelo -ne "") { $comp += @("--suelo", $Suelo) }
  & docker @comp | Out-Host
  Log "Hecho. Manda: $Salida\resultado_128_c5.json"
}
$rcFases = 0
if ($Fases) { $rcFases = [int](PasoFases | Select-Object -Last 1) }
Limpiar
exit $rcFases
