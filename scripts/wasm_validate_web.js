#!/usr/bin/env node
/**
 * Validate an ONNX model using onnxruntime-web with the WASM execution
 * provider — the exact runtime used by browsers loading the bundle.
 *
 * This script is intentionally separate from wasm_validate.js (which uses
 * onnxruntime-node with native binaries). Here the inference path is:
 *   ONNX model → onnxruntime-web → WebAssembly → numeric output
 *
 * Usage:
 *   node wasm_validate_web.js <model.onnx> <spec.json>
 *
 * spec.json: same format as wasm_validate.js
 *   { input_name, input_dtype, input_data, expected_output_shape? }
 *
 * Exit 0 + JSON result on success, exit 1 on failure.
 */

const ort = require('onnxruntime-web');
const fs = require('fs');
const path = require('path');

// Point ORT Web at its WASM binaries (bundled alongside the npm package)
const wasmDir = path.join(__dirname, '..', 'node_modules', 'onnxruntime-web', 'dist');
ort.env.wasm.wasmPaths = wasmDir + path.sep;

async function main() {
  const [, , modelPath, specPath] = process.argv;
  if (!modelPath || !specPath) {
    process.stderr.write('Usage: node wasm_validate_web.js <model.onnx> <spec.json>\n');
    process.exit(1);
  }

  const spec = JSON.parse(fs.readFileSync(specPath, 'utf8'));
  const { input_name, input_dtype, input_data, expected_output_shape } = spec;

  const session = await ort.InferenceSession.create(
    modelPath,
    { executionProviders: ['wasm'] }
  );

  const results = [];
  for (const row of input_data) {
    const flat = row.flat ? row.flat() : row;
    const shape = [1, flat.length];
    let tensor;
    if (input_dtype === 'int64') {
      tensor = new ort.Tensor('int64', BigInt64Array.from(flat.map(BigInt)), shape);
    } else {
      tensor = new ort.Tensor('float32', Float32Array.from(flat), shape);
    }
    const feeds = { [input_name]: tensor };
    const output = await session.run(feeds);
    const outputKeys = Object.keys(output);
    const outData = Array.from(output[outputKeys[0]].data);
    const outDims = Array.from(output[outputKeys[0]].dims);
    results.push({ data: outData, dims: outDims });

    if (expected_output_shape) {
      const match = expected_output_shape.every((d, i) => d < 0 || d === outDims[i]);
      if (!match) {
        process.stderr.write(
          `Shape mismatch: expected ${JSON.stringify(expected_output_shape)}, got ${JSON.stringify(outDims)}\n`
        );
        process.exit(1);
      }
    }
  }

  process.stdout.write(JSON.stringify({
    ok: true,
    backend: 'wasm',
    runtime: 'onnxruntime-web',
    n_samples: results.length,
    results,
  }));
  process.exit(0);
}

main().catch(err => {
  process.stderr.write(`Error: ${err.message || err}\n`);
  process.exit(1);
});
