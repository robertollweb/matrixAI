#!/usr/bin/env node
/**
 * Validate an ONNX model using onnxruntime-node (same ONNX runtime used by
 * onnxruntime-web). Reads inference parameters from a JSON spec file and
 * verifies the model loads and runs without error.
 *
 * Usage:
 *   node wasm_validate.js <model.onnx> <spec.json>
 *
 * spec.json format:
 *   {
 *     "input_name": "Input",
 *     "input_dtype": "int64" | "float32",
 *     "input_data": [[1, 2, 3, ...]],   // array of arrays (batch x length)
 *     "expected_output_shape": [1, 2]   // optional shape check
 *   }
 *
 * Exit 0 on success, 1 on failure (prints error to stderr).
 */

const ort = require('onnxruntime-node');
const fs = require('fs');

async function main() {
  const [, , modelPath, specPath] = process.argv;
  if (!modelPath || !specPath) {
    process.stderr.write('Usage: node wasm_validate.js <model.onnx> <spec.json>\n');
    process.exit(1);
  }

  const spec = JSON.parse(fs.readFileSync(specPath, 'utf8'));
  const { input_name, input_dtype, input_data, expected_output_shape } = spec;

  const session = await ort.InferenceSession.create(modelPath);

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
    const outDims = output[outputKeys[0]].dims;
    results.push({ data: outData, dims: outDims });

    if (expected_output_shape) {
      const match = expected_output_shape.every((d, i) => d < 0 || d === outDims[i]);
      if (!match) {
        process.stderr.write(
          `Shape mismatch: expected ${JSON.stringify(expected_output_shape)}, got ${JSON.stringify(Array.from(outDims))}\n`
        );
        process.exit(1);
      }
    }
  }

  process.stdout.write(JSON.stringify({ ok: true, n_samples: results.length, results }));
  process.exit(0);
}

main().catch(err => {
  process.stderr.write(`Error: ${err.message}\n`);
  process.exit(1);
});
