import contextlib
import io
import unittest
import tempfile
from pathlib import Path
import shutil

from matrixai.parameters import build_initial_parameter_set, write_parameter_set
from matrixai.pack import pack_model
from matrixai.parser import parse_file

ROOT = Path(__file__).resolve().parents[1]


class TestPack(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.out_dir = Path(self.temp_dir) / "dist"
        
        self.model_path = ROOT / "examples" / "email-agent.typed.mxai"
        self.params_path = Path(self.temp_dir) / "params.json"
        parameter_set = build_initial_parameter_set(parse_file(self.model_path))
        write_parameter_set(self.params_path, parameter_set)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_pack_model_without_docker(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = pack_model(self.model_path, None, docker=False, outdir=self.out_dir)
        self.assertEqual(result, 0)

        self.assertTrue((self.out_dir / "email-agent.typed.mxai").exists())
        self.assertTrue((self.out_dir / "matrixai").exists())
        self.assertTrue((self.out_dir / "matrixai" / "__init__.py").exists())

        self.assertFalse((self.out_dir / "Dockerfile").exists())

    def test_pack_model_with_docker_and_params(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = pack_model(self.model_path, self.params_path, docker=True, outdir=self.out_dir)
        self.assertEqual(result, 0)

        self.assertTrue((self.out_dir / "email-agent.typed.mxai").exists())
        self.assertTrue((self.out_dir / "params.json").exists())

        dockerfile_path = self.out_dir / "Dockerfile"
        self.assertTrue(dockerfile_path.exists())
        content = dockerfile_path.read_text()

        self.assertIn("COPY email-agent.typed.mxai /app/", content)
        self.assertIn("COPY params.json /app/", content)
        self.assertIn("HEALTHCHECK", content)
        self.assertIn('CMD ["python3", "-m", "matrixai", "serve", "email-agent.typed.mxai", "--params", "params.json"', content)
        # docker-compose.yml and .env.example must also be generated
        self.assertTrue((self.out_dir / "docker-compose.yml").exists())
        self.assertTrue((self.out_dir / ".env.example").exists())

    def test_pack_model_rejects_invalid_params(self):
        bad_params = Path(self.temp_dir) / "bad_params.json"
        bad_params.write_text(self.params_path.read_text().replace("mxai_", "mxai_bad_", 1))

        with contextlib.redirect_stdout(io.StringIO()):
            result = pack_model(self.model_path, bad_params, docker=True, outdir=self.out_dir)

        self.assertEqual(result, 1)
        self.assertFalse(self.out_dir.exists())


if __name__ == "__main__":
    unittest.main()
