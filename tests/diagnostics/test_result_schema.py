import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

from graphfl_lab.config_io import add_config_argument, parse_args_with_config
from graphfl_lab.diagnostics.result_schema import (
    LEGACY_RESULT_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    result_schema_version,
    validate_result_schema,
    with_result_schema,
)


class ResultSchemaTest(unittest.TestCase):
    def test_with_result_schema_adds_required_fields(self):
        payload = with_result_schema({"meta": {}, "results": {}})

        self.assertEqual(payload["result_schema_version"], RESULT_SCHEMA_VERSION)
        self.assertEqual(payload["config_aliases_used"], [])
        self.assertEqual(payload["unsupported_components"], [])
        self.assertEqual(validate_result_schema(payload), [])

    def test_missing_version_reads_as_v0(self):
        self.assertEqual(result_schema_version({"meta": {}}), LEGACY_RESULT_SCHEMA_VERSION)

    def test_config_aliases_are_recorded(self):
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

        self.assertEqual(args._config_aliases_used, ("spectral_filter_strength->graph_filter_strength",))


if __name__ == "__main__":
    unittest.main()
