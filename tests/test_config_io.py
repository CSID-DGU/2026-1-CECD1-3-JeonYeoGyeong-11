import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

from spectral_fl.config_io import add_config_argument, parse_args_with_config


class ConfigIoTest(unittest.TestCase):
    def test_config_overrides_parser_defaults(self):
        parser = argparse.ArgumentParser()
        add_config_argument(parser)
        parser.add_argument("--rounds", type=int, default=30)
        parser.add_argument("--diagnostic-only", default=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(
                json.dumps({"args": {"rounds": 2, "diagnostic_only": True}}),
                encoding="utf-8",
            )
            old_argv = sys.argv
            try:
                sys.argv = ["prog", "--config", str(path)]
                args = parse_args_with_config(parser)
            finally:
                sys.argv = old_argv

        self.assertEqual(args.rounds, 2)
        self.assertTrue(args.diagnostic_only)

    def test_cli_overrides_config_defaults(self):
        parser = argparse.ArgumentParser()
        add_config_argument(parser)
        parser.add_argument("--rounds", type=int, default=30)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(json.dumps({"args": {"rounds": 2}}), encoding="utf-8")
            old_argv = sys.argv
            try:
                sys.argv = ["prog", "--config", str(path), "--rounds", "5"]
                args = parse_args_with_config(parser)
            finally:
                sys.argv = old_argv

        self.assertEqual(args.rounds, 5)


if __name__ == "__main__":
    unittest.main()
