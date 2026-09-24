# -*- coding: utf-8 -*-
"""문서 대 코드 대조 검사.

    python -m commerce.tools.gate docs

기존 게이트는 코드 대 코드만 본다. 이 검사는 문서가 약속한 것과 코드에 실제로
있는 것이 어긋나는 경우를 잡는다. 판정은 종료 코드 0과 마지막 줄
`DOCS OK: links=<n> env=<n> gates=<n> imports=<n> vendor=<n> fences=<n> failures=0`의
동시 충족이다.

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
CONTRACTS_DOC = REPO_ROOT / "docs/contracts.md"
DEVELOPMENT_DOC = REPO_ROOT / "docs/development.md"

# 옛 비공개 문서명. 공개 전환 뒤 본문에 남으면 독자가 없는 파일을 찾게 된다.
LEGACY_NAMES = (
    "START_HERE", "WORKING_AGREEMENT", "ARCHITECTURE", "CONTRACTS",
    "MODEL_BOUNDARY", "LAB_WORKFLOW", "GIT_WORKFLOW", "COMPARISON", "EVALUATION",
)
# docs/README.md는 이전 문서명 매핑표, docs/contracts.md는 그 매핑 설명, start.md §5는
# 역할 카드 독자용 약어표로 옛 이름을 알아보게 하려고 적는다.
LEGACY_EXEMPT = {"docs/README.md", "docs/contracts.md", "docs/team/start.md"}
# 게이트가 출력하는 판정 문자열이며 문서명이 아니다.
LEGACY_ALLOWED_PHRASE = "CONTRACTS OK"

# 도구마다 읽는 지시 파일이 다르다. AGENTS.md만 본문이고 나머지는 포인터여야
# 에이전트마다 다른 버전의 규칙을 읽는 일이 생기지 않는다.
VENDOR_INSTRUCTION_FILES = (
    "CLAUDE.md", "GEMINI.md", "GROK.md", ".cursorrules", ".windsurfrules",
    ".github/copilot-instructions.md", ".aider.conf.yml", ".junie/guidelines.md",
)
VENDOR_POINTER_MAX_LINES = 12

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)\s]+?)(?:#[^)]*)?\)")
# 링크 문법 밖에 적힌 상대 문서 경로. 렌더하면 눌리지 않고 링크 검사도 비켜 간다.
BARE_DOC_PATH_RE = re.compile(r"(?<![\w(/.])(\.\./[\w./-]+\.md)")
# CommonMark 코드 펜스. 닫는 펜스 뒤에는 공백만 올 수 있다.
FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
ENV_TOKEN_RE = re.compile(r"\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b")
# 서비스 README의 환경변수 절. 담당이 자기 파일에 적으므로 보호 문서를 고치지 않고 값을 추가한다.
ENV_LINE_PREFIX = "현재 코드가 읽는 값:"


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
        for number, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for match in BARE_DOC_PATH_RE.finditer(LINK_RE.sub("", line)):
                fail(_rel(doc), "링크가 아닌 문서 경로",
                     "%s:%d %s ([이름](경로) 형식으로 쓴다)" % (_rel(doc), number, match.group(1)))
    return checked


# --- 1b. 코드 펜스: 원문은 멀쩡해도 GitHub 렌더가 깨지는 경우 ------------------

def _check_fences(fail) -> int:
    """닫는 펜스 뒤에 글자가 붙으면 CommonMark는 닫힌 것으로 보지 않는다.

    그러면 파일 끝까지 전부 코드 블록으로 렌더되고 그 안의 링크도 눌리지 않는다.
    에이전트는 원문을 읽으므로 알아채지 못하고 사람만 깨진 화면을 본다.
    """
    checked = 0
    for doc in _docs():
        opened = None  # (문자, 길이, 줄 번호)
        for number, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            match = FENCE_RE.match(line)
            if not match:
                continue
            run, rest = match.group(1), match.group(2)
            if opened is None:
                if run[0] == "`" and "`" in rest:
                    continue  # ```x``` 같은 인라인 코드는 펜스가 아니다.
                opened = (run[0], len(run), number)
                checked += 1
            elif run[0] == opened[0] and len(run) >= opened[1]:
                if rest.strip():
                    fail(_rel(doc), "닫는 펜스 뒤에 본문이 붙음",
                         "%s:%d 펜스 다음 줄로 본문을 내린다" % (_rel(doc), number))
                # 렌더러는 여기서 닫지 않지만, 한 번 보고했으니 뒤쪽 펜스까지 연쇄로 틀리지 않게 닫는다.
                opened = None
        if opened is not None:
            fail(_rel(doc), "닫히지 않은 코드 펜스", "%s:%d" % (_rel(doc), opened[2]))
    return checked


# --- 2. 환경변수: 서비스 README의 '현재 코드가 읽는 값' vs 코드 ----------------

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


def _documented_env(service_dir: pathlib.Path) -> Set[str]:
    """서비스 README에서 '현재 코드가 읽는 값:' 줄의 키를 읽는다."""
    readme = service_dir / "README.md"
    for line in readme.read_text(encoding="utf-8").splitlines():
        if ENV_LINE_PREFIX in line:
            return set(ENV_TOKEN_RE.findall(line.split(ENV_LINE_PREFIX, 1)[1]))
    raise LookupError("%s에 '%s' 줄이 없다" % (_rel(readme), ENV_LINE_PREFIX))


def _service_env_reads(service_dir: pathlib.Path) -> Tuple[Set[str], Set[str], Dict[str, str]]:
    """서비스 디렉터리의 모든 .py가 읽는 키. main.py만 보면 다른 모듈에서 읽는 키를 놓친다."""
    required: Set[str] = set()
    optional: Set[str] = set()
    where: Dict[str, str] = {}
    for source in sorted(service_dir.rglob("*.py")):
        req, opt = _env_reads(source)
        required |= req
        optional |= opt
        for key in req | opt:
            where.setdefault(key, _rel(source))
    return required, optional, where


def _check_env(fail) -> int:
    services = {
        name: REPO_ROOT / "commerce/services" / name
        for name in ("central_api", "merchant_api", "fl_coordinator")
    }
    checked = 0
    for service, service_dir in services.items():
        readme = _rel(service_dir / "README.md")
        try:
            listed = _documented_env(service_dir)
        except LookupError as exc:
            fail(readme, "환경변수 절 없음", str(exc))
            continue
        required, optional, where = _service_env_reads(service_dir)
        actual = required | optional
        checked += len(actual | listed)
        for key in sorted(actual - listed):
            fail(readme, "코드가 읽는데 README에 없음", "%s (%s)" % (key, where[key]))
        for key in sorted(listed - actual):
            fail(readme, "README의 '현재' 값인데 코드가 읽지 않음", "%s (%s)" % (key, _rel(service_dir)))

    # 런처는 자식 프로세스의 환경을 비우고 자기 dict에 적은 값만 넘긴다. 그래서
    # 어느 서비스든 필수 키를 새로 만들면서 런처가 넘기지 않으면 기동이 깨진다.
    launcher_keys = _env_written_by_launcher()
    for service, service_dir in services.items():
        required, _, where = _service_env_reads(service_dir)
        for key in sorted(required - launcher_keys):
            fail("commerce/deploy/run_local.py", "필수 환경변수를 런처가 넘기지 않음",
                 "%s (%s)" % (key, where[key]))
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

def _check_vendor_pointers(fail) -> int:
    """도구별 지시 파일은 AGENTS.md를 가리키는 짧은 포인터여야 한다."""
    checked = 0
    if not (REPO_ROOT / "AGENTS.md").is_file():
        fail("AGENTS.md", "공통 지시 파일이 없음", "모든 에이전트의 기준 문서다")
        return checked
    for name in VENDOR_INSTRUCTION_FILES:
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        checked += 1
        text = path.read_text(encoding="utf-8")
        lines = [line for line in text.splitlines() if line.strip()]
        if "AGENTS.md" not in text:
            fail(name, "AGENTS.md를 가리키지 않음",
                 "도구별 파일만 읽는 에이전트가 규칙을 못 본다")
        if len(lines) > VENDOR_POINTER_MAX_LINES:
            fail(name, "포인터가 아니라 규칙을 담고 있음",
                 "본문 %d줄. AGENTS.md로 옮기고 %d줄 이하로 유지하라"
                 % (len(lines), VENDOR_POINTER_MAX_LINES))
    return checked


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
    vendor = _check_vendor_pointers(fail)
    fences = _check_fences(fail)
    _check_legacy_names(fail)

    print("문서 대 코드 대조: 링크 %d·환경변수 %d·게이트 %d·import %d·도구지시 %d·코드펜스 %d, 실패 %d건."
          % (links, env, gates, imports, vendor, fences, len(failures)))
    if failures:
        # contracts 러너와 같은 규약: 실패 시 OK 줄을 내지 않는다.
        for where, kind, detail in failures:
            print("  %-42s %-34s %s" % (where, kind, detail))
        return 1
    if verbose:
        print("문서 %d개를 읽었다." % len(_docs()))
    print("DOCS OK: links=%d env=%d gates=%d imports=%d vendor=%d fences=%d failures=0"
          % (links, env, gates, imports, vendor, fences))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
