# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""116-C5 — prueba de que las particiones del KIT son idénticas (mismos índices) a las de la pasada 116-C2.

Dos modos, cada uno en su PROCESO (para que no se mezclen los módulos):

    python3 comprobar_particiones.py huella --raiz <dir con matrixAI/ y matrixai-engines/> --home <HOME con fase0_openml_datos/>
    python3 comprobar_particiones.py comparar            # corre los dos y compara: el árbol de trabajo y el kit

`huella` escribe, por conjunto del kit y por pliegue, el sha256 de (entrena, valida, test) además del `plan.digest()` y el
`pliegues.digest()`. `comparar` lanza la huella del ÁRBOL (el código y los ARFF que usó 116-C2) y la del KIT
(`codigo/` + los ARFF desempaquetados de `datos/`, con `python3 -E` y la ruta del kit PRIMERO; el host no tiene las dependencias sin el usuario (`-s`), así que la prueba sin árbol de verdad es la de la imagen; el editable resuelve `matrixai` desde
cualquier sitio) y exige que sean iguales, pliegue a pliegue, en TODOS los conjuntos.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent


def huella(raiz: Path, home: Path) -> dict:
    os.environ["HOME"] = str(home)
    fase0 = raiz / "matrixAI" / "benchmarks" / "fase0"
    for r in (fase0, raiz / "matrixai-engines" / "src", raiz / "matrixAI"):
        sys.path.insert(0, str(r))
    import pasada_116c2_tabicl as c2  # noqa: PLC0415
    import pasada_exploratoria_101_c3 as c3  # noqa: PLC0415
    import matrixai  # noqa: PLC0415
    c2.preparar_protocolo_v2()
    protocolo = c3.protocolo_registrado()
    seleccion = json.loads((AQUI / "conjuntos.json").read_text(encoding="utf-8"))
    nombres = [f["dataset"] for f in seleccion["incluidos"]]
    por_nombre = {d.nombre: d for d in c2.c5.datasets_de_la_pasada(protocolo)}
    salida = {"matrixai_importado_de": matrixai.__file__, "arff_dir": str(c3.ARFF_DIR), "conjuntos": {}}
    for nombre in nombres:
        ds = por_nombre[nombre]
        folds, n_cols, n_test, base = c2.folds_del_dataset(ds, protocolo)
        por_id, propuesta, *_ = base
        test_ids = propuesta.plan.observaciones_del_rol("test")
        pliegues = []
        for repeticion in range(protocolo.particion.repeticiones_para(ds.cubo)):
            for i in range(protocolo.particion.folds):
                p = propuesta.pliegues.pliegue_de(repeticion=repeticion, pliegue=i)
                if p is None:
                    continue
                h = hashlib.sha256(json.dumps([list(p.entrena), list(p.valida), list(test_ids)]).encode()).hexdigest()
                pliegues.append([repeticion, i, len(p.entrena), len(p.valida), h])
        salida["conjuntos"][nombre] = {"plan_digest": propuesta.plan.digest(), "pliegues_digest": propuesta.pliegues.digest(),
                                       "n_test": len(test_ids), "semillas": list(protocolo.particion.semillas), "pliegues": pliegues}
    return salida


def _desempaquetar(home: Path) -> None:
    destino = home / "fase0_openml_datos" / "arff"
    destino.mkdir(parents=True, exist_ok=True)
    manifiesto = json.loads((AQUI / "datos" / "manifiesto.json").read_text(encoding="utf-8"))
    for data_id, m in manifiesto.items():
        with gzip.open(AQUI / "datos" / f"{data_id}.arff.gz", "rb") as g, (destino / f"{data_id}.arff").open("wb") as f:
            shutil.copyfileobj(g, f)
        assert hashlib.sha256((destino / f"{data_id}.arff").read_bytes()).hexdigest() == m["sha256_del_arff"], data_id


def comparar() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        home_kit = Path(tmp) / "home_kit"
        _desempaquetar(home_kit)
        arbol = subprocess.run([sys.executable, __file__, "huella", "--raiz", "/home/deployer", "--home", str(Path.home())],
                               capture_output=True, text=True, check=True, cwd="/")
        kit = subprocess.run([sys.executable, "-E", __file__, "huella", "--raiz", str(AQUI / "codigo"),
                              "--home", str(home_kit)], capture_output=True, text=True, check=True, cwd="/",
                             env={"PATH": os.environ.get("PATH", "")})
    a, k = json.loads(arbol.stdout.splitlines()[-1]), json.loads(kit.stdout.splitlines()[-1])
    print("árbol importa matrixai de:", a["matrixai_importado_de"], "| ARFF:", a["arff_dir"])
    print("kit   importa matrixai de:", k["matrixai_importado_de"], "| ARFF:", k["arff_dir"])
    assert str(AQUI / "codigo") in k["matrixai_importado_de"], "el kit NO está usando su propio código"
    malos = 0
    for nombre, ca in a["conjuntos"].items():
        ck = k["conjuntos"][nombre]
        igual = ca == ck
        malos += not igual
        print(f"  {'IGUAL' if igual else 'DISTINTO'} {nombre}: {len(ca['pliegues'])} pliegues, plan {ca['plan_digest'][:12]}, "
              f"pliegues {ca['pliegues_digest'][:12]}")
    print(f"{len(a['conjuntos']) - malos} de {len(a['conjuntos'])} conjuntos con las particiones idénticas (índices de entrena/valida/test)")
    return 1 if malos else 0


if __name__ == "__main__":
    if sys.argv[1] == "huella":
        args = dict(zip(sys.argv[2::2], sys.argv[3::2]))
        print(json.dumps(huella(Path(args["--raiz"]), Path(args["--home"]))))
    else:
        raise SystemExit(comparar())
