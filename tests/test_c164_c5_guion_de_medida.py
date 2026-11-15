"""164-C5: la lógica PROPIA del guion de medida (sin lanzar un solo estudio): el payload que construye es el de la persona y lo que lee
del sobre es lo que el sobre dice, con lo ausente como `None` y nunca como cero."""
import csv
import io
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "benchmarks" / "c164"))
import medir_c164_c5 as m  # noqa: E402


def test_el_payload_lleva_el_forecast_de_cada_ejemplo_y_solo_las_columnas_que_declara():
    filas = [{"borough": "Bronx", "pickup_dt": "2015-01-01 01:00:00", "pickups": "5", "temp": "30.0", "hday": "Y", "extra": "x"}]
    p = m.payload_de("uber_h6", filas, 3, "completo")
    assert p["forecast"] == {"time_column": "pickup_dt", "series_column": "borough", "horizon": 6, "lags": 24,
                             "lag_columns": ["temp"], "season": 24, "extra_inputs": ["hday"]}
    cab = next(csv.reader(io.StringIO(p["csv_text"])))
    assert cab == ["pickup_dt", "borough", "pickups", "temp", "hday"] and "modo" not in p      # completo = sin `modo`: el camino de siempre
    assert p["objetivo"] == "pickups" and p["folds"] == 3 and "tarea" not in p
    assert m.payload_de("avocado", [], 3, "rapido")["modo"] == "rapido"
    assert "extra_inputs" not in m.payload_de("lluvia", [], 3, "completo")["forecast"]


def test_el_resumen_lee_del_sobre_y_lo_ausente_es_none_no_cero():
    sobre = {"pronostico": {"horizonte_pasos": 1, "limites": [{"clave": "intervalos_con_filas_solapadas"}],
                            "nota": {"prueba": {"n": 10, "modelo": {"mae": 2.0}, "persistencia": {"mae": 4.0}, "estacional": None,
                                                "mejora_mae_sobre_persistencia": 0.5, "mejora_mae_sobre_estacional": None, "mase": 0.8},
                                     "pliegues": [{"pliegue": 0, "n": 3, "mejora_sobre_persistencia": -0.1,
                                                   "mejora_sobre_estacional": None}],
                                     "veredicto_por_pliegue": {"persistencia": {"medidos": 1, "gana": 0, "pierde": 1, "empata": 0}}}},
             "conforme": {"calibracion": {"alfa": 0.1, "cuantil": 3.0, "n_calibracion": 50},
                          "cobertura_en_test": {"cobertura_observada": 0.88, "n": 10}}}
    r = m.resumen_del_sobre(sobre, {"seleccion": {"candidate_engine": "lightgbm", "decision": {"outcome": "selected"}}})
    assert r["campeon"] == "lightgbm" and r["prueba"]["mae_estacional"] is None and r["prueba"]["mejora_mae_sobre_estacional"] is None
    assert r["prueba"]["mejora_mae_sobre_persistencia"] == 0.5 and r["pliegues"][0]["mejora_sobre_estacional"] is None
    assert r["intervalo"]["cobertura_nominal"] == 0.9 and r["intervalo"]["cobertura_medida_en_la_prueba"]["cobertura_observada"] == 0.88
    assert r["limites"] == ["intervalos_con_filas_solapadas"]
    assert "n/d" in m.lectura_en_una_linea({**r, "prueba": {**r["prueba"], "mejora_mae_sobre_estacional": None}})


def test_sin_calibracion_el_intervalo_no_esta_disponible_y_sin_prueba_no_hay_prueba():
    r = m.resumen_del_sobre({"pronostico": {"nota": {"prueba": None, "motivo": {"es": "x", "en": "y"}}},
                             "conforme": {"calibracion": None, "cobertura_en_test": None}}, {})
    assert r["prueba"] is None and r["intervalo"]["disponible"] is False and r["intervalo"]["cobertura_nominal"] is None


def test_la_lluvia_se_queda_con_los_ultimos_dias_de_cada_ciudad_y_descarta_las_cortas(tmp_path):
    cab = "temp_max,temp_min,temp_mean,rain,wind_speed,pressure,pressure_delta,humidity,cloud_cover,dew_point,precip_hours,wind_dir_sin,wind_dir_cos,lat,lon,elevation,llueve_manana"
    def fila(lat, rain):
        return f"1,1,1,{rain},1,1000,0,50,10,1,0,0,0,{lat},0,0,0"
    ruta = tmp_path / "l.csv"
    ruta.write_text("\n".join([cab] + [fila(1.0, i) for i in range(5)] + [fila(2.0, 9)] + [fila(3.0, i) for i in range(10, 13)]) + "\n")
    filas, info = m.cargar_lluvia(ruta, dias=3)
    assert info["ciudades"] == 2 and info["ciudades_con_menos_dias_descartadas"] == 1      # la de 1 fila no entra
    c0 = [f["rain"] for f in filas if f["ciudad"] == "C0"]
    assert c0 == ["2", "3", "4"]                                                          # los ÚLTIMOS 3 días, en orden
    assert {f["dia"] for f in filas if f["ciudad"] == "C0"} == {f["dia"] for f in filas if f["ciudad"] == "C1"}
