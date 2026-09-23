# -*- coding: utf-8 -*-
"""계약 스키마와 fixture 검증 러너.

기준: CONTRACTS.md §8(오류 코드), §9.3(배열 중복), §10.1(디렉터리 규약 X1~X6), MODEL_BOUNDARY.md §4.
표준 라이브러리 + jsonschema만 사용한다.

팀이 인용하는 정본 명령은 저장소 루트에서 실행하는 다음 하나다(CONTRACTS.md §8 X4).
    python -m commerce.tools.gate contracts

이 파일은 그 게이트가 감싸 호출하는 contracts 패키지 내부 명령이며 단독 실행도 같은 판정을 낸다.
    python -m commerce.packages.contracts.validate
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover
    sys.stderr.write("jsonschema가 필요하다. `pip install jsonschema` 후 다시 실행한다.\n")
    raise SystemExit(2)

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_DIR = os.path.join(HERE, "schemas")
FIXTURE_DIR = os.path.join(HERE, "fixtures")

# CONTRACTS.md §8 표의 순서. E2에 따라 이 순서의 첫 번째 코드 하나만 돌려준다.
ERROR_CODE_ORDER: Tuple[str, ...] = (
    "SCHEMA_INVALID",
    "UNKNOWN_FIELD",
    "MISSING_REQUIRED_FIELD",
    "INVALID_TYPE",
    "INVALID_ENUM_VALUE",
    "VERSION_MISMATCH",
    "DUPLICATE_EVENT",
    "DUPLICATE_IDEMPOTENCY_KEY",
    "ILLEGAL_STATE_TRANSITION",
    "STALE_STATUS_VERSION",
    "FORBIDDEN",
    "NOT_FOUND",
    # B-C 경계 전용(MODEL_BOUNDARY.md §7 검사 8행). CONTRACTS.md §8 표와 같은 순서다.
    "MANIFEST_MISMATCH",
    "TENSOR_SET_MISMATCH",
    "DUPLICATE_ROUND_SUBMIT",
    "ROUND_DISCARDED",
)
CODE_RANK = {code: i for i, code in enumerate(ERROR_CODE_ORDER)}

# CONTRACTS.md §8 X3이 정본이다. enum·const-enum 필드가 없는 계약은 INVALID_ENUM_VALUE
# 사례를 만들 수 없으므로 VERSION_MISMATCH 사례로 대체한다. 목록을 바꾸려면 X3을 먼저 고친다.
X3_REQUIRED = ("UNKNOWN_FIELD", "MISSING_REQUIRED_FIELD", "INVALID_ENUM_VALUE")
X3_NO_ENUM_CONTRACTS = {"recommendation_request.v1", "round_config.v1", "model_release.v1"}
X3_SUBSTITUTE = "VERSION_MISMATCH"

CONDITIONAL_KEYWORDS = {"allOf", "anyOf", "if", "then", "else", "not"}


# --------------------------------------------------------------------------- 분류

def _path_str(parts: Sequence[Any]) -> str:
    """RFC 6901 JSON Pointer 문자열."""
    if not parts:
        return ""
    out = []
    for p in parts:
        token = str(p).replace("~", "~0").replace("/", "~1")
        out.append(token)
    return "/" + "/".join(out)


def _classify(err) -> List[Tuple[str, str]]:
    """jsonschema 오류 1건을 (code, field_path) 목록으로 옮긴다."""
    schema_path = list(err.schema_path)
    field_path = _path_str(list(err.absolute_path))

    if err.validator in ("oneOf", "anyOf"):
        if err.context:
            branches: Dict[int, List[Any]] = {}
            for sub in err.context:
                branches.setdefault(sub.schema_path[0], []).append(sub)
            best = min(branches.values(), key=len)
            out: List[Tuple[str, str]] = []
            for sub in best:
                out.extend(_classify(sub))
            return out
        return [("SCHEMA_INVALID", field_path)]

    # 조건부 제약(if/then/else, allOf) 위반은 CONTRACTS.md §8의 SCHEMA_INVALID다.
    if CONDITIONAL_KEYWORDS.intersection(schema_path):
        return [("SCHEMA_INVALID", field_path)]

    if err.validator == "additionalProperties":
        return [("UNKNOWN_FIELD", field_path)]
    if err.validator == "required":
        return [("MISSING_REQUIRED_FIELD", field_path)]
    if err.validator == "type":
        return [("INVALID_TYPE", field_path)]
    if err.validator == "enum":
        return [("INVALID_ENUM_VALUE", field_path)]
    if err.validator == "const":
        if list(err.absolute_path)[-1:] == ["schema_version"]:
            return [("VERSION_MISMATCH", field_path)]
        return [("INVALID_ENUM_VALUE", field_path)]
    return [("SCHEMA_INVALID", field_path)]


# --------------------------------------------------------------------------- 의미 규칙

def _dup(values: Sequence[str]) -> Optional[str]:
    seen: Set[str] = set()
    for v in values:
        if v in seen:
            return v
        seen.add(v)
    return None


def _semantic_errors(contract: str, payload: Any) -> List[Tuple[str, str, str]]:
    """JSON Schema로 표현할 수 없는 계약 규칙. (code, field_path, 설명)."""
    out: List[Tuple[str, str, str]] = []
    if not isinstance(payload, dict):
        return out

    if contract == "round_submission.v1":
        for code, path, detail in _semantic_errors(
                "delta_manifest.v1", payload.get("delta_manifest")):
            out.append((code, "/delta_manifest" + path, detail))

    if contract in ("commerce_order.v1", "purchase_event.v1"):
        items = payload.get("items") or []
        ids = [i.get("item_id_local") for i in items if isinstance(i, dict)]
        d = _dup(ids)
        if d is not None:
            out.append(("SCHEMA_INVALID", "/items",
                        "items 안에서 item_id_local이 중복됐다"))

    if contract == "catalog_snapshot.v1":
        items = payload.get("items") or []
        ids = [i.get("item_id_local") for i in items if isinstance(i, dict)]
        d = _dup(ids)
        if d is not None:
            out.append(("SCHEMA_INVALID", "/items",
                        "items 안에서 item_id_local이 중복됐다"))

    if contract == "recommendation.v1":
        items = payload.get("items") or []
        ids = [i.get("item_id_local") for i in items if isinstance(i, dict)]
        if _dup(ids) is not None:
            # CONTRACTS.md SC4
            out.append(("DUPLICATE_EVENT", "/items",
                        "응답 안에서 item_id_local이 중복됐다(SC4)"))
        keys = [(-float(i["score"]), str(i["item_id_local"])) for i in items
                if isinstance(i, dict) and "score" in i and "item_id_local" in i]
        for a, b in zip(keys, keys[1:]):
            if a > b:
                # CONTRACTS.md SC3
                out.append(("SCHEMA_INVALID", "/items",
                            "items가 score 내림차순·item_id_local 사전순 정렬이 아니다(SC3)"))
                break

    if contract == "recommendation_request.v1":
        cand = payload.get("candidate_item_ids")
        if isinstance(cand, list) and _dup([c for c in cand if isinstance(c, str)]) is not None:
            # CONTRACTS.md §8. 스키마의 uniqueItems가 먼저 잡지만 의미 검사로도 못 박는다.
            out.append(("SCHEMA_INVALID", "/candidate_item_ids",
                        "candidate_item_ids 안에서 item_id_local이 중복됐다"))

    if contract == "round_submit_ack.v1":
        # CONTRACTS.md §6. 과거 확장 enum도 형태 검증은 유지한다. 집계 처분이면 들어간 라운드를 반드시 밝히고,
        # 접수·대기·폐기 처분이면 아직(또는 영영) 들어간 라운드가 없다.
        disp = payload.get("disposition")
        agg_in = payload.get("aggregated_in_round_id")
        aggregated = disp in ("aggregated_on_time", "aggregated_carried")
        if aggregated and agg_in is None:
            out.append(("SCHEMA_INVALID", "/aggregated_in_round_id",
                        "disposition이 집계인데 aggregated_in_round_id가 null이다"))
        if (disp is not None) and (not aggregated) and agg_in is not None:
            out.append(("SCHEMA_INVALID", "/aggregated_in_round_id",
                        "disposition이 집계가 아닌데 aggregated_in_round_id가 non-null이다"))

    if contract in ("shared_model_manifest.v1", "delta_manifest.v1"):
        tensors = payload.get("tensors") or []
        names = [t.get("name") for t in tensors if isinstance(t, dict)]
        if _dup(names) is not None:
            out.append(("SCHEMA_INVALID", "/tensors",
                        "tensors 안에서 name이 중복됐다"))
        for idx, name in enumerate(names):
            if isinstance(name, str) and (name.startswith("local.")
                                          or name.startswith("frozen_text.")):
                # MODEL_BOUNDARY.md §3 전송 금지 목록
                out.append(("SCHEMA_INVALID", "/tensors/%d/name" % idx,
                            "local.* 또는 frozen_text.* 텐서는 manifest·delta에 싣지 않는다(공유 경계)"))
                break

    return out


# --------------------------------------------------------------------------- 검증

class Verdict:
    def __init__(self, ok: bool, code: Optional[str], field_path: Optional[str],
                 detail: Optional[str]) -> None:
        self.ok = ok
        self.code = code
        self.field_path = field_path
        self.detail = detail


def validate_payload(validator: Draft202012Validator, contract: str, payload: Any) -> Verdict:
    """payload 1건을 검증한다. 거부면 CONTRACTS.md §8의 code 하나를 낸다."""
    found: List[Tuple[str, str, str]] = []
    def check_finite(value: Any, path: List[Any]) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            found.append(("SCHEMA_INVALID", _path_str(path), "Non-finite number"))
        elif isinstance(value, dict):
            for key, child in value.items():
                check_finite(child, path + [key])
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                check_finite(child, path + [idx])

    # JSON exponents can overflow the parser's float even without an Infinity literal.
    check_finite(payload, [])
    for err in validator.iter_errors(payload):
        for code, path in _classify(err):
            found.append((code, path, err.message))

    if not found:
        # 스키마를 통과한 payload에만 의미 규칙을 건다.
        found = _semantic_errors(contract, payload)

    if not found:
        return Verdict(True, None, None, None)

    found.sort(key=lambda t: CODE_RANK.get(t[0], len(CODE_RANK)))
    code, path, detail = found[0]
    return Verdict(False, code, path, detail)


class NonJsonConstant(ValueError):
    """RFC 8259 밖의 NaN·Infinity·-Infinity 리터럴. CONTRACTS.md SC1이 금지한다."""


def _reject_constant(token: str):
    raise NonJsonConstant(token)


def load_payload(path: str) -> Tuple[Any, Optional[Verdict]]:
    """fixture 1건을 엄격하게 읽는다. NaN·Infinity 리터럴은 SCHEMA_INVALID로 거부한다."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        return json.loads(text, parse_constant=_reject_constant), None
    except NonJsonConstant as exc:
        return None, Verdict(False, "SCHEMA_INVALID", None,
                             "RFC 8259 밖의 리터럴 %s 다. score 등 number 필드에 NaN·Infinity를 쓰지 않는다(SC1)"
                             % exc.args[0])
    except json.JSONDecodeError as exc:
        return None, Verdict(False, "SCHEMA_INVALID", None, "JSON 파싱 실패: %s" % exc)


def load_schemas() -> Dict[str, Draft202012Validator]:
    out: Dict[str, Draft202012Validator] = {}
    if not os.path.isdir(SCHEMA_DIR):
        raise SystemExit("schemas/ 디렉터리가 없다: %s" % SCHEMA_DIR)
    for fname in sorted(os.listdir(SCHEMA_DIR)):
        if not fname.endswith(".schema.json"):
            continue
        contract = fname[: -len(".schema.json")]
        with open(os.path.join(SCHEMA_DIR, fname), "r", encoding="utf-8") as f:
            schema = json.load(f)
        Draft202012Validator.check_schema(schema)
        out[contract] = Draft202012Validator(schema)
    return out


def read_reason(path: str) -> Tuple[Optional[str], str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines()]
    if not lines or not lines[0].strip():
        return None, ""
    return lines[0].strip(), " ".join(ln.strip() for ln in lines[1:]).strip()


def run(verbose: bool = False) -> int:
    validators = load_schemas()
    failures: List[str] = []
    passed = 0
    checked_contracts: List[str] = []

    # The local $defs copy keeps validation offline; reject schema drift.
    if "round_submission.v1" in validators and "delta_manifest.v1" in validators:
        expected = dict(validators["delta_manifest.v1"].schema)
        expected.pop("$schema", None)
        expected.pop("$id", None)
        embedded = validators["round_submission.v1"].schema.get("$defs", {}).get("delta_manifest")
        if embedded != expected:
            failures.append("round_submission.v1 delta_manifest definition differs from its source schema")

    if not os.path.isdir(FIXTURE_DIR):
        failures.append("fixtures/ 디렉터리가 없다: %s" % FIXTURE_DIR)
        fixture_contracts: List[str] = []
    else:
        fixture_contracts = sorted(d for d in os.listdir(FIXTURE_DIR)
                                   if os.path.isdir(os.path.join(FIXTURE_DIR, d)))

    for contract in sorted(validators):
        if contract not in fixture_contracts:
            failures.append("[%s] fixtures/%s/ 가 없다" % (contract, contract))
    for contract in fixture_contracts:
        if contract not in validators:
            failures.append("[%s] schemas/%s.schema.json 이 없다" % (contract, contract))

    for contract in fixture_contracts:
        if contract not in validators:
            continue
        checked_contracts.append(contract)
        validator = validators[contract]
        base = os.path.join(FIXTURE_DIR, contract)

        valid_dir = os.path.join(base, "valid")
        valid_files = sorted(f for f in os.listdir(valid_dir)
                             if f.endswith(".json")) if os.path.isdir(valid_dir) else []
        if len(valid_files) < 2:
            failures.append("[%s] valid fixture가 %d개다. 2개 이상이어야 한다"
                            % (contract, len(valid_files)))
        for fname in valid_files:
            fpath = os.path.join(valid_dir, fname)
            payload, verdict = load_payload(fpath)
            if verdict is None:
                verdict = validate_payload(validator, contract, payload)
            if verdict.ok:
                passed += 1
                if verbose:
                    print("  PASS valid   %s/%s" % (contract, fname))
            else:
                failures.append("[%s] valid/%s 가 %s 로 거부됐다 (%s / %s)"
                                % (contract, fname, verdict.code, verdict.field_path,
                                   verdict.detail))

        invalid_dir = os.path.join(base, "invalid")
        invalid_files = sorted(f for f in os.listdir(invalid_dir)
                               if f.endswith(".json")) if os.path.isdir(invalid_dir) else []
        if len(invalid_files) < 3:
            failures.append("[%s] invalid fixture가 %d개다. 3개 이상이어야 한다"
                            % (contract, len(invalid_files)))

        seen_codes: Set[str] = set()
        for fname in invalid_files:
            fpath = os.path.join(invalid_dir, fname)
            rpath = fpath[: -len(".json")] + ".reason.txt"
            if not os.path.isfile(rpath):
                failures.append("[%s] invalid/%s 에 대응하는 .reason.txt 가 없다 (X1)"
                                % (contract, fname))
                continue
            expected_code, note = read_reason(rpath)
            if expected_code is None:
                failures.append("[%s] invalid/%s 의 .reason.txt 첫 줄이 비어 있다"
                                % (contract, fname))
                continue
            if expected_code not in CODE_RANK:
                failures.append("[%s] invalid/%s 의 code %s 가 CONTRACTS.md §8 표에 없다"
                                % (contract, fname, expected_code))
                continue
            if not note:
                failures.append("[%s] invalid/%s 의 .reason.txt 에 거부 사유 한 줄이 없다"
                                % (contract, fname))
            seen_codes.add(expected_code)
            payload, verdict = load_payload(fpath)
            if verdict is None:
                verdict = validate_payload(validator, contract, payload)
            if verdict.ok:
                failures.append("[%s] invalid/%s 가 통과했다. 기대 code %s"
                                % (contract, fname, expected_code))
            elif verdict.code != expected_code:
                failures.append("[%s] invalid/%s 의 code가 다르다. 기대 %s, 실제 %s (%s / %s)"
                                % (contract, fname, expected_code, verdict.code,
                                   verdict.field_path, verdict.detail))
            else:
                passed += 1
                if verbose:
                    print("  PASS invalid %s/%s -> %s" % (contract, fname, verdict.code))

        # X1 역방향: 짝 없는 .reason.txt
        if os.path.isdir(invalid_dir):
            for fname in sorted(os.listdir(invalid_dir)):
                if fname.endswith(".reason.txt"):
                    jname = fname[: -len(".reason.txt")] + ".json"
                    if not os.path.isfile(os.path.join(invalid_dir, jname)):
                        failures.append("[%s] invalid/%s 에 대응하는 .json 이 없다 (X1)"
                                        % (contract, fname))

        # X3
        required = set(X3_REQUIRED)
        if contract in X3_NO_ENUM_CONTRACTS:
            required.discard("INVALID_ENUM_VALUE")
            required.add(X3_SUBSTITUTE)
        missing = sorted(required - seen_codes)
        if missing:
            failures.append("[%s] X3 미달. 없는 거부 사례: %s" % (contract, ", ".join(missing)))

    print("계약 %d개, fixture 판정 %d건 통과, 실패 %d건."
          % (len(checked_contracts), passed, len(failures)))
    if failures:
        print("")
        for line in failures:
            print("FAIL " + line)
        return 1
    # 판정 문자열. 통과 시 마지막 줄로 이 ASCII 한 줄만 낸다.
    # 기준: WORKING_AGREEMENT.md §4 게이트 표 contracts 행·§6 G1 기대 출력.
    # schema와 fixture 개수는 실행 결과로 보고한다.
    print("CONTRACTS OK: schemas=%d fixtures=%d failures=0"
          % (len(checked_contracts), passed))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="contracts.validate",
        description="commerce/packages/contracts 의 스키마와 fixture를 검증한다.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="fixture 1건마다 판정을 출력한다")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run(verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
