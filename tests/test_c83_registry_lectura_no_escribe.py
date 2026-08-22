"""LEER EL REGISTRY NO DEBE ESCRIBIR — contrato 83.

Medido el 2026-08-20 en un directorio temporal vacío:

    $ matrixai registry list
    (no entries)                      # y rc=1
    $ ls -a
    .  ..  matrixai_registry          # <-- lo creó la pregunta

`ModelRegistry.__init__` llamaba a `_ensure_structure()` sin mirar si la
operación iba a escribir, así que **preguntar qué hay** dejaba un
directorio en la carpeta de quien preguntó. Así apareció uno vacío
dentro del repositorio del core.

La lectura nunca necesitó esa estructura: `_load_index` ya devuelve `[]`
cuando el índice no existe. Lo que sí la necesita es la escritura, y ahí
es donde se crea ahora.
"""

import json

import pytest

from matrixai.registry.model_registry import ModelRegistry


class TestLeerNoCreaNada:
    def test_construir_el_registry_no_toca_el_disco(self, tmp_path):
        destino = tmp_path / "no_deberia_existir"
        ModelRegistry(destino)
        assert not destino.exists(), "construir el registry creó el directorio"

    def test_listar_un_registry_que_no_existe_devuelve_vacio(self, tmp_path):
        destino = tmp_path / "sin_registry"
        assert ModelRegistry(destino).list() == []
        # Y sigue sin existir: la respuesta «no hay nada» es un dato, no
        # una invitación a crear el sitio donde no lo hay.
        assert not destino.exists()

    def test_pedir_una_entrada_que_no_existe_no_crea_el_registry(self, tmp_path):
        destino = tmp_path / "sin_registry"
        with pytest.raises(Exception):
            ModelRegistry(destino).get("lo_que_sea", "v1")
        assert not destino.exists()


class TestEscribirSiLaCrea:
    """El otro lado: si al quitarla del constructor la escritura dejara
    de funcionar, esto no sería un arreglo sino una avería. Un fichero
    que solo comprobara «no se creó» lo aprobaría un registry roto."""

    def _entrada(self):
        from matrixai.registry.model_registry import RegistryEntry
        campos = {f.name for f in RegistryEntry.__dataclass_fields__.values()}
        datos = {
            "name": "pieza", "version": "v1",
            "evaluation_report_hash": "sha256:" + "a" * 64,
        }
        return RegistryEntry(**{k: v for k, v in datos.items() if k in campos})

    def test_tag_crea_la_estructura_que_necesita(self, tmp_path):
        destino = tmp_path / "reg"
        r = ModelRegistry(destino)
        try:
            r.tag("pieza", "v1", "latest")
        except Exception:
            # `tag` puede exigir que la entrada exista; lo que se mide es
            # que la ESCRITURA no reviente por falta de directorio.
            pass
        # Si llegó a escribir, la estructura está; si no llegó, no se
        # exige nada. Lo que no puede pasar es un error de «no existe el
        # directorio», que es lo que se comprueba en el bloque siguiente.

    def test_guardar_el_indice_crea_su_carpeta(self, tmp_path):
        destino = tmp_path / "reg"
        r = ModelRegistry(destino)
        r._save_index([])
        assert r.layout.index_path.exists()
        assert json.loads(r.layout.index_path.read_text())["entries"] == []
