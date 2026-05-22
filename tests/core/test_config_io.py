import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

from graphfl_lab.config_io import add_config_argument, load_config, parse_args_with_config, resolve_config_path


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
        self.assertIn("rounds", args._config_arg_dests)

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
        self.assertIn("rounds", args._explicit_arg_dests)
        self.assertIn("rounds", args._user_arg_dests)

    def test_general_config_path_aliases_to_vision_tree(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vision_path = root / "configs" / "vision" / "diagnostic" / "smoke.json"
            old_path = root / "configs" / "general" / "diagnostic" / "smoke.json"
            vision_path.parent.mkdir(parents=True)
            vision_path.write_text(json.dumps({"args": {"rounds": 3}}), encoding="utf-8")

            self.assertEqual(resolve_config_path(old_path), vision_path)
            self.assertEqual(load_config(old_path)["rounds"], 3)

    def test_spectral_filter_strength_config_aliases_to_graph_key(self):
        parser = argparse.ArgumentParser()
        add_config_argument(parser)
        parser.add_argument(
            "--graph-filter-strength",
            dest="graph_filter_strength",
            type=float,
            default=1.0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(
                json.dumps({"args": {"spectral_filter_strength": 0.5}}),
                encoding="utf-8",
            )
            old_argv = sys.argv
            try:
                sys.argv = ["prog", "--config", str(path)]
                args = parse_args_with_config(parser)
            finally:
                sys.argv = old_argv

        self.assertEqual(args.graph_filter_strength, 0.5)
        self.assertFalse(hasattr(args, "spectral_filter_strength"))

    def test_graph_filter_strength_config_populates_compat_attr(self):
        parser = argparse.ArgumentParser()
        add_config_argument(parser)
        parser.add_argument(
            "--graph-filter-strength",
            dest="graph_filter_strength",
            type=float,
            default=1.0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(
                json.dumps({"args": {"graph_filter_strength": 0.25}}),
                encoding="utf-8",
            )
            old_argv = sys.argv
            try:
                sys.argv = ["prog", "--config", str(path)]
                args = parse_args_with_config(parser)
            finally:
                sys.argv = old_argv

        self.assertEqual(args.graph_filter_strength, 0.25)


if __name__ == "__main__":
    unittest.main()
