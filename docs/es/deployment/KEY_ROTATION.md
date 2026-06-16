# PR4-C4 — Gestión y Rotación de Signing Keys

> **English:** [docs/en/deployment/KEY_ROTATION.md](../../en/deployment/KEY_ROTATION.md)

MatrixAI usa dos claves HMAC-SHA256 independientes:

| Clave | Variable de entorno | Protege |
|---|---|---|
| Clave de acción | `MATRIXAI_ACTION_SIGNING_KEY` | Firmas de `ActionTrace` (P20) |
| Clave del registry | `MATRIXAI_REGISTRY_SIGNING_KEY` | Firmas de entradas del registry (P21) |

**Garantía crítica:** rotar una clave nunca invalida trazas o entradas del registry firmadas con la clave anterior. El fichero de historial conserva todas las claves retiradas para que la verificación histórica sea posible indefinidamente.

---

## Generar una clave segura

```bash
openssl rand -hex 32
# ej.: 4a7f3b2c9e1d6f8a...  (64 caracteres hex = 256 bits)
```

La longitud mínima aceptada es 32 bytes (64 caracteres hex). Valores más cortos son aceptados por el runtime pero no se recomiendan para producción.

---

## Configuración inicial

Establece ambas claves en `.env` antes del primer despliegue:

```
MATRIXAI_ACTION_SIGNING_KEY=<openssl rand -hex 32>
MATRIXAI_REGISTRY_SIGNING_KEY=<openssl rand -hex 32>
```

Si `MATRIXAI_REGISTRY_SIGNING_KEY` no está establecida, el registry genera una clave local por contenedor (almacenada en `matrixai_registry/.registry_signing_key`, permisos 0600). Esta clave local no persiste entre reemplazos de contenedor — establece la variable de entorno explícitamente para producción.

---

## Rotar una clave

La rotación es un procedimiento de tres pasos:

### Paso 1 — Registrar la clave saliente en el historial

```bash
# Rotar la clave de acción
matrixai keys rotate \
  --purpose action \
  --registry-path matrixai_registry

# Rotar la clave del registry
matrixai keys rotate \
  --purpose registry \
  --registry-path matrixai_registry
```

Esto marca la clave actual con `rotated_at` en el fichero de historial (`matrixai_registry/.matrixai_key_history.json`). El valor de la clave se conserva para verificación histórica.

**Haz esto antes de cambiar la variable de entorno.** Si cambias la variable primero y luego registras la clave vieja, debes pasar `--key <valor-clave-vieja>` explícitamente.

### Paso 2 — Establecer la nueva clave

```bash
# Generar nueva clave
NEW_KEY=$(openssl rand -hex 32)
echo $NEW_KEY  # copia este valor

# Actualizar .env
# MATRIXAI_ACTION_SIGNING_KEY=<nuevo valor>
```

### Paso 3 — Reiniciar el servidor

```bash
docker compose restart
```

Las nuevas firmas usan la nueva clave inmediatamente. Las trazas históricas siguen siendo verificables.

---

## Verificar firmas históricas

Después de la rotación, las trazas antiguas pueden verificarse de dos formas:

**1. Clave explícita (cuando sabes qué clave firmó la traza):**

```bash
matrixai audit-action \
  --trace trace.json \
  --signing-key <valor-clave-vieja>
```

**2. Vía key store (fallback automático a través de todas las claves conocidas):**

```python
from matrixai.signing.keystore import KeyStore
from matrixai.actions.trace import verify_action_trace_with_keystore

store = KeyStore.load(Path("matrixai_registry/.matrixai_key_history.json"))
verified = store.try_verify_action(trace, current_key)
```

Para entradas del registry (que almacenan `signing_key_fingerprint`), el key store busca la clave por fingerprint directamente:

```python
verified = store.try_verify_registry_entry(
    entry_hash, signature, fingerprint, current_key
)
```

---

## Listar el historial de claves

```bash
matrixai keys list --registry-path matrixai_registry

# Key history: matrixai_registry/.matrixai_key_history.json
# Fingerprint              Purpose     Status      Added                        Rotated
# sha256:e861b2eab679927c  action      retired     2026-05-27T12:00:00+00:00    2026-05-27T13:00:00+00:00
# sha256:4a7f3b2c9e1d6f8a  action      active      2026-05-27T13:00:00+00:00    -

# Salida JSON
matrixai keys list --registry-path matrixai_registry --json
```

---

## Qué pasa si una clave se pierde o se compromete

### Clave perdida (no comprometida)

Si la clave se pierde pero no está comprometida:

- Las trazas existentes firmadas con esa clave **ya no pueden verificarse criptográficamente**. La firma está intacta pero la clave de verificación no existe.
- Las nuevas firmas usan la clave actual.
- **Mitigación:** Mantén el fichero de historial respaldado junto al registry. Ambos viven en `matrixai_registry/` por defecto.

Lo que se preserva incluso sin la clave perdida:
- La cadena de `entry_hash` y `model_hash` — son hashes SHA256 de los artefactos, no de la clave. La integridad del contenido de los artefactos puede seguir comprobándose recalculando el hash.
- La detección de tampering vía `registry verify` — rehashea los ficheros almacenados contra los hashes del manifest, independientemente de la signing key.

### Clave comprometida

Si una signing key está comprometida:

1. **Rotar inmediatamente** (Pasos 1-3 de arriba).
2. La clave comprometida sigue en el historial. Un atacante con la clave podría falsificar firmas — pero **solo para nuevos artefactos falsificados**; no puede cambiar retroactivamente la cadena de `entry_hash` (que depende del contenido de los artefactos, no de la clave).
3. Considera la clave comprometida como "no fiable para nuevas firmas" — las trazas existentes firmadas antes del compromiso no se ven afectadas en términos de integridad del contenido.
4. Si el fichero de historial también está en riesgo, elimina la entrada de la clave comprometida manualmente (edita el JSON y elimina la entrada con el fingerprint comprometido).

---

## Formato del fichero de historial

El fichero de historial es un array JSON en `matrixai_registry/.matrixai_key_history.json` (permisos 0600):

```json
[
  {
    "fingerprint": "sha256:e861b2eab679927c",
    "key": "abc123def456...",
    "purpose": "action",
    "added_at": "2026-05-27T12:00:00+00:00",
    "rotated_at": "2026-05-27T13:00:00+00:00"
  },
  {
    "fingerprint": "sha256:4a7f3b2c9e1d6f8a",
    "key": "nuevovaolrclave...",
    "purpose": "action",
    "added_at": "2026-05-27T13:00:00+00:00",
    "rotated_at": null
  }
]
```

`rotated_at: null` significa que la clave está actualmente activa. Un valor no nulo significa que la clave ha sido retirada pero se conserva para verificación histórica.

Sobrescribe la ruta por defecto con `MATRIXAI_KEY_HISTORY_PATH=/ruta/al/key_history.json`.

---

## Notas de seguridad

- El fichero de historial almacena valores de clave en texto plano. Protégelo como cualquier secreto: permisos de sistema de ficheros 0600, excluido del control de versiones, respaldado cifrado.
- El modelo de amenaza es integridad en un host confiable. Un atacante con acceso de lectura al sistema de ficheros podría leer claves históricas. Si esa amenaza es relevante, usa un gestor de secretos (Vault, AWS Secrets Manager) y pasa las claves solo via variable de entorno — nunca las escribas en el fichero de historial en esa configuración.
- Los fingerprints (`sha256:<16hex>`) son resúmenes cortos de la clave, seguros para loguear y mostrar. No revelan el valor de la clave.
