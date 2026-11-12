# Selección de los conjuntos de CONFIRMACIÓN del 162-C1: la regla de benchmarks/c160/CONFIRMACION.md, tomando los SIGUIENTES a
# los 6 del 160. SOLO METADATOS de OpenML (nombre, licencia, filas, columnas, clases). Ningún modelo.
import json, os, sys, urllib.request, time
from pathlib import Path
RAIZ = Path('/home/deployer/matrixAI/benchmarks/fase0')
def get(u):
    for i in range(4):
        try:
            return json.load(urllib.request.urlopen(u, timeout=60))
        except Exception as e:
            time.sleep(2)
    raise SystemExit(f'sin respuesta: {u}')
sys.path.insert(0, str(RAIZ))
import pasada_exploratoria_101_c3 as c3, pasada_amplia_101_c5 as c5
_ds = c5.datasets_de_la_pasada(c3.protocolo_registrado())  # el catalogo del protocolo registrado (40; incluye los 8 SELLADOS)
_declarados = json.loads((RAIZ/'resultado_pasada_119_c3.json').read_text())['datasets_declarados']  # los 32 medidos por el 119-C3
_sellados = [d for d in _ds if d.sellado]
assert len(_ds) == 40 and len(_declarados) == 32 and len(_sellados) == 8
# La definicion del 160 = los 32 + los 8 sellados. UNION conservadora con el catalogo del protocolo (que tambien trae breast-w,
# adult y elevators): no cambia la eleccion salvo que excluye breast-w, que ya es del 160.
if os.environ.get('SOLO_DEFINICION_DEL_160'):
    ids_f0 = {int(x['data_id']) for x in _declarados} | {int(d.data_id) for d in _sellados}
    nombres_f0 = {x['nombre'].lower() for x in _declarados} | {d.nombre.lower() for d in _sellados}
else:
    ids_f0 = {int(x['data_id']) for x in _declarados} | {int(d.data_id) for d in _ds}
    nombres_f0 = {x['nombre'].lower() for x in _declarados} | {d.nombre.lower() for d in _ds}
# Hermano de origen de un conjunto de la Fase 0 (clausula 6 ampliada, solo metadatos): satimage (182) es el Statlog Landsat Satellite
# de UCI; 'Satellite' (40900, SELLADO) es su variante de deteccion de anomalias (Goldstein) del mismo Statlog Landsat.
HERMANOS = {182: 'hermano de origen de Satellite (40900, sellado): Statlog Landsat'}
del_160 = {15, 29, 23, 54, 44956, 44957, 31, 44, 188, 307, 44958, 44959}  # los 6 del 160 y los 6 del 162
LIC = {'public', 'cc by 4.0', 'cc by', 'cc0', 'cc 0: public domain'}
def estudio(n):
    d = get(f'https://openml.org/api/v1/json/study/{n}')['study']
    return sorted(int(i) for i in d['data']['data_id'])
def meta(i):
    d = get(f'https://openml.org/api/v1/json/data/{i}')['data_set_description']
    q = get(f'https://openml.org/api/v1/json/data/qualities/{i}')['data_qualities']['quality']
    q = {x['name']: x['value'] for x in q}
    return d, q
log, elegidos = [], []
for estudio_id, tipos in ((99, ('binaria', 'multiclase')), (353, ('regresion',))):
    cupo = {t: 0 for t in tipos}
    for i in estudio(estudio_id):
        if all(c >= 2 for c in cupo.values()):
            break
        d, q = meta(i)
        nombre = d['name']; lic = (d.get('licence') or '').strip()
        n = float(q.get('NumberOfInstances', 0)); f = float(q.get('NumberOfFeatures', 0)); k = float(q.get('NumberOfClasses', 0) or 0)
        motivo = None
        if i in ids_f0 or nombre.lower() in nombres_f0: motivo = 'ya en la Fase 0'
        elif i in HERMANOS: motivo = HERMANOS[i]
        elif i in del_160: motivo = 'es de los 6 del 160'
        elif lic.lower() not in LIC: motivo = f'licencia {lic!r}'
        elif not (500 <= n <= 2500): motivo = f'filas {n:.0f} (la palanca solo actua con train < 2000)'
        elif f > 101: motivo = f'columnas {f:.0f}'
        elif nombre.lower().startswith('mfeat-'): motivo = 'hermano mfeat-'
        elif (float(q.get('NumberOfSymbolicFeatures', 0)) - (1 if estudio_id == 99 else 0)) < 3: motivo = f'predictores nominales {float(q.get("NumberOfSymbolicFeatures", 0)) - (1 if estudio_id == 99 else 0):.0f} < 3'
        if estudio_id == 99:
            t = 'binaria' if k == 2 else ('multiclase' if 3 <= k <= 20 else None)
            if t is None and motivo is None: motivo = f'clases {k:.0f}'
        else:
            t = 'regresion'
        if motivo is None and cupo.get(t, 2) >= 2:
            motivo = f'cupo de {t} lleno'
        log.append({'estudio': estudio_id, 'id': i, 'nombre': nombre, 'licencia': lic, 'filas': n, 'cols': f, 'clases': k, 'descartado': motivo, 'tipo': t})
        if motivo is None:
            cupo[t] += 1
            elegidos.append({'tarea': t, 'id': i, 'nombre': nombre, 'version': d.get('version'), 'licencia': lic, 'filas': int(n), 'cols': int(f), 'clases': int(k) or None, 'faltantes': int(float(q.get('NumberOfMissingValues', 0))), 'md5': d.get('md5_checksum'), 'url': d.get('url'), 'formato': d.get('format')})
Path(os.environ.get('SELECCION_LOG', 'seleccion_log.json')).write_text(json.dumps({'elegidos': elegidos, 'log': log}, indent=1, ensure_ascii=False))
for e in elegidos: print(e)
print(len(log), 'candidatos mirados')
