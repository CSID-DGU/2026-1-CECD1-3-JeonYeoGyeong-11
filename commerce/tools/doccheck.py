# -*- coding: utf-8 -*-
"""문서 대 코드 대조 검사.

    python -m commerce.tools.gate docs

기존 게이트는 코드 대 코드만 본다. 이 검사는 문서가 약속한 것과 코드에 실제로
있는 것이 어긋나는 경우를 잡는다. 판정은 종료 코드 0과 마지막 줄
`DOCS OK: links=<n> env=<n> gates=<n> imports=<n> failures=0`의 동시 충족이다.

표준 라이브러리만 사용한다. 실패는 파일·항목·기대/실제를 함께 출력한다.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import pathlib
import re
from typing import Dict, Iterable, List, Set, Tuple

HERE = pathlib.Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
WORKING_AGREEMENT = REPO_ROOT / "docs/team/working-agreement.md"
CONTRACTS_DOC = REPO_ROOT / "docs/contracts.md"
DEVELOPMENT_DOC = REPO_ROOT / "docs/development.md"

# 옛 비공개 문서명. 공개 전환 뒤 본문에 남으면 독자가 없는 파일을 찾게 된다.
LEGACY_NAMES = (
    "START_HERE", "WORKING_AGREEMENT", "ARCHITECTURE", "CONTRACTS",
    "MODEL_BOUNDARY", "LAB_WORKFLOW", "GIT_WORKFLOW", "COMPARISON", "EVALUATION",
)
# docs/README.md는 이전 문서명 매핑표, docs/contracts.md는 그 매핑 설명을 담는다.
LEGACY_EXEMPT = {"docs/README.md", "docs/contracts.md"}
# 게이트가 출력하는 판정 문자열이며 문서명이 아니다.
LEGACY_ALLOWED_PHRASE = "CONTRACTS OK"

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)\s]+?)(?:#[^)]*)?\)")
ENV_TOKEN_RE = re.compile(r"\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b")
MODULE_CELL_RE = re.compile(r"commerce\.services\.([a-z_]+)\.main:app")


def _docs() -> List[pathlib.Path]:
    found = sorted((REPO_ROOT / "docs").rglob("*.md"))
    found += [REPO_ROOT / "README.md", REPO_ROOT / "AGENTS.md", REPO_ROOT / "CLAUDE.md"]
    found += sorted((REPO_ROOT / "commerce").rglob("README.md"))
    return [p for p in found if p.is_file()]


def _rel(path: pathlib.Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


# --- 1. 상대 링크 -------------------------------------------------------------

def _check_links(fail) -> int:
    checked = 0
    for doc in _docs():
        for match in LINK_RE.finditer(doc.read_text(encoding="utf-8")):
            target = match.group(1)
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            checked += 1
            if not (doc.parent / target).resolve().exists():
                fail(_rel(doc), "끊어진 링크", target)
    return checked


# --- 2. 환경변수: 문서의 '현재 코드가 읽는 값' vs 코드 ------------------------

def _env_reads(path: pathlib.Path) -> Tuple[Set[str], Set[str]]:
    """(필수 키, 선택 키). os.environ[...]는 필수, os.environ.get(...)은 선택."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    required: Set[str] = set()
    optional: Set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
                and node.value.attr == "environ" and isinstance(node.slice, ast.Constant)):
            required.add(node.slice.value)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "environ" and node.args
                and isinstance(node.args[0], ast.Constant)):
            optional.add(node.args[0].value)
    return required, optional


def _env_written_by_launcher() -> Set[str]:
    """run_local.py가 자식 프로세스에 넘기는 키."""
    tree = ast.parse((REPO_ROOT / "commerce/deploy/run_local.py").read_text(encoding="utf-8"))
    keys: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str) \
                        and ENV_TOKEN_RE.fullmatch(key.value):
                    keys.add(key.value)
    return keys


def _documented_env() -> Dict[str, Set[str]]:
    """working-agreement §3 표에서 서비스별 '현재 코드가 읽는 값'을 읽는다.

    열은 이름으로 찾는다. 열 순서가 바뀌어도 깨지지 않는다.
    """
    rows = [line for line in WORKING_AGREEMENT.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("|")]
    header_index = None
    current_col = None
    for position, line in enumerate(rows):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if "현재 코드가 읽는 값" in cells:
            header_index = position
            current_col = cells.index("현재 코드가 읽는 값")
            break
    if header_index is None:
        raise LookupError("working-agreement §3에서 '현재 코드가 읽는 값' 열을 찾지 못했다")

    documented: Dict[str, Set[str]] = {}
    for line in rows[header_index + 1:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) <= current_col:
            continue
        service = None
        for cell in cells:
            found = MODULE_CELL_RE.search(cell)
            if found:
                service = found.group(1)
                break
        if service is None:
            continue
        documented[service] = set(ENV_TOKEN_RE.findall(cells[current_col]))
    return documented


def _check_env(fail) -> int:
    services = {
        "central_api": REPO_ROOT / "commerce/services/central_api/main.py",
        "merchant_api": REPO_ROOT / "commerce/services/merchant_api/main.py",
        "fl_coordinator": REPO_ROOT / "commerce/services/fl_coordinator/main.py",
    }
    try:
        documented = _documented_env()
    except LookupError as exc:
        fail(_rel(WORKING_AGREEMENT), "표 해석 실패", str(exc))
        return 0

    checked = 0
    for service, source in services.items():
        required, optional = _env_reads(source)
        actual = required | optional
        listed = documented.get(service, set())
        checked += len(actual | listed)
        for key in sorted(actual - listed):
            fail(_rel(WORKING_AGREEMENT), "코드가 읽는데 표에 없음",
                 "%s (%s)" % (key, _rel(source)))
        for key in sorted(listed - actual):
            fail(_rel(WORKING_AGREEMENT), "표의 '현재' 값인데 코드가 읽지 않음",
                 "%s (%s)" % (key, _rel(source)))

    # 런처가 값을 넘기지 않으면 필수 키를 새로 만든 순간 기동이 깨진다.
    launcher_keys = _env_written_by_launcher()
    merchant_required, _ = _env_reads(services["merchant_api"])
    for key in sorted(merchant_required - launcher_keys):
        fail("commerce/deploy/run_local.py", "필수 환경변수를 런처가 넘기지 않음", key)
    return checked


# --- 3. 게이트 이름: docs/contracts.md 표 vs GATES 레지스트리 -----------------

def _check_gates(fail) -> int:
    from commerce.tools.gate import GATES
    text = CONTRACTS_DOC.read_text(encoding="utf-8")
    listed = set()
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        first = line.strip().strip("|").split("|")[0].strip()
        name = first.split(" ")[0]
        if re.fullmatch(r"[a-z][a-z0-9]*", name) and name not in ("이름", "게이트"):
            listed.add(name)
    for name in sorted(listed - set(GATES)):
        fail(_rel(CONTRACTS_DOC), "표에 있으나 GATES에 없는 게이트", name)
    for name in sorted(set(GATES) - listed):
        fail(_rel(CONTRACTS_DOC), "GATES에 있으나 표에 없는 게이트", name)
    return len(listed | set(GATES))


# --- 4. 문서에 적힌 import 가 실제로 되는가 -----------------------------------

IMPORT_RE = re.compile(r"^from (commerce[\w.]*) import ([\w, ]+)$", re.M)


def _check_imports(fail) -> int:
    checked = 0
    for doc in _docs():
        for module_name, names in IMPORT_RE.findall(doc.read_text(encoding="utf-8")):
            checked += 1
            if importlib.util.find_spec(module_name) is None:
                fail(_rel(doc), "문서의 import 모듈 없음", module_name)
                continue
            module = importlib.import_module(module_name)
            for name in [n.strip() for n in names.split(",") if n.strip()]:
                if not hasattr(module, name):
                    fail(_rel(doc), "문서의 import 이름 없음", "%s.%s" % (module_name, name))
    return checked


# --- 5. 옛 비공개 문서명이 본문에 남아 있는가 ---------------------------------

def _check_legacy_names(fail) -> None:
    pattern = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(LEGACY_NAMES) + r")(?![A-Za-z0-9_])")
    for doc in _docs():
        if _rel(doc) in LEGACY_EXEMPT:
            continue
        for number, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for match in pattern.finditer(line):
                if line[match.start():].startswith(LEGACY_ALLOWED_PHRASE):
                    continue
                fail(_rel(doc), "옛 문서명이 링크 없이 남음",
                     "%s:%d %s" % (_rel(doc), number, match.group(1)))


# --- 진입점 -------------------------------------------------------------------

def run(verbose: bool = False) -> int:
    failures: List[Tuple[str, str, str]] = []

    def fail(where: str, kind: str, detail: str) -> None:
        failures.append((where, kind, detail))

    links = _check_links(fail)
    env = _check_env(fail)
    gates = _check_gates(fail)
    imports = _check_imports(fail)
    _check_legacy_names(fail)

    print("문서 대 코드 대조: 링크 %d·환경변수 %d·게이트 %d·import %d, 실패 %d건."
          % (links, env, gates, imports, len(failures)))
    if failures:
        # contracts 러너와 같은 규약: 실패 시 OK 줄을 내지 않는다.
        for where, kind, detail in failures:
            print("  %-42s %-34s %s" % (where, kind, detail))
        return 1
    if verbose:
        print("문서 %d개를 읽었다." % len(_docs()))
    print("DOCS OK: links=%d env=%d gates=%d imports=%d failures=0"
          % (links, env, gates, imports))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
