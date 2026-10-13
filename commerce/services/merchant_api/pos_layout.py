"""Each store's own floor plan: zones, tables (named the way the staff call them), fixtures.

A plan is a set of zones (홀, 룸, 테라스, 2층 ...) and, in each, items on a COLS x ROWS grid. A
table has a name ("3", "A2", "룸1", "바4"), seats and a shape; a fixture (kitchen, counter, door,
window ...) only helps the staff find their way on the POS screen. The owner draws the plan in
the editor (/seller/{id}/floor) and the POS draws the same plan.

A table's item_no is its stable id and what a ticket stores as table_no: renaming a table,
moving it, or moving it to another zone keeps its open check and its past receipts. A table
removed from the plan is only deactivated (its past checks still point at it), and a table that
has an open check cannot be removed.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from commerce.services.merchant_api.pos_db import row, rows

COLS, ROWS = 24, 14
TABLE_SHAPES = {"square": "사각", "rect": "직사각", "round": "원형", "booth": "부스", "bar": "바 좌석", "floor": "좌식"}
FIXTURES = {"kitchen": "주방", "counter": "카운터", "entrance": "입구", "restroom": "화장실", "window": "창",
            "wall": "벽", "selfbar": "셀프바", "stairs": "계단"}
MAX_ZONES, MAX_ITEMS = 8, 200


class LayoutError(ValueError):
    """A plan the editor sent that cannot be saved; the message is shown to the owner as is."""


def label(name: Optional[str], long: bool = False) -> str:
    """How the screens call a table: "3" is "3번" ("3번 테이블"), "A2" is "A2" ("A2 테이블"), "룸1" stays."""
    if not name:
        return "테이블"
    if name.isdigit():
        return name + ("번 테이블" if long else "번")
    if long and re.fullmatch(r"[A-Za-z]+-?\d+", name):
        return name + " 테이블"
    return name


def default_plan(tables: int) -> dict[str, Any]:
    """A plain hall of `tables` four-seat tables in rows, for a store that never drew its plan."""
    items, per_row = [], 6
    for i in range(min(tables, 36)):
        r, c = divmod(i, per_row)
        items.append({"kind": "table", "zone": "홀", "name": str(i + 1), "seats": 4, "shape": "square",
                      "x": 1 + c * 4, "y": 1 + r * 3, "w": 3, "h": 2})
    if len(items) <= 24:
        items.append({"kind": "fixture", "zone": "홀", "name": "카운터", "seats": 0, "shape": "counter",
                      "x": 1, "y": ROWS - 2, "w": 5, "h": 1})
        items.append({"kind": "fixture", "zone": "홀", "name": "입구", "seats": 0, "shape": "entrance",
                      "x": COLS - 4, "y": ROWS - 1, "w": 3, "h": 1})
    return {"zones": ["홀"], "items": items}


def ensure(conn, seller_id: str) -> None:
    """A store without a plan gets the default one (from the old table count), once."""
    if row(conn, "SELECT 1 FROM pos_floor WHERE seller_id = ? LIMIT 1", (seller_id,)):
        return
    s = row(conn, "SELECT tables FROM pos_settings WHERE seller_id = ?", (seller_id,))
    _write(conn, seller_id, _validate(default_plan((s or {}).get("tables") or 12)), {})


def install(conn, seller_id: str, payload: Any) -> bool:
    """Put a first plan in place for a store that has none yet (demo data); False if it had one."""
    if row(conn, "SELECT 1 FROM pos_floor WHERE seller_id = ? LIMIT 1", (seller_id,)):
        return False
    _write(conn, seller_id, _validate(payload), {})
    return True


def plan(conn, seller_id: str) -> dict[str, Any]:
    ensure(conn, seller_id)
    items = rows(conn, "SELECT item_no AS no, kind, zone, name, seats, shape, x, y, w, h FROM pos_floor "
                       "WHERE seller_id = ? AND active = 1 ORDER BY zone, y, x", (seller_id,))
    s = row(conn, "SELECT zones FROM pos_settings WHERE seller_id = ?", (seller_id,))
    zones = json.loads(s["zones"]) if s and s.get("zones") else []
    for i in items:  # a zone an item names but the list lost is still shown
        if i["zone"] not in zones:
            zones.append(i["zone"])
    return {"zones": zones or ["홀"], "items": items, "cols": COLS, "rows": ROWS}


def tables(conn, seller_id: str) -> list[dict[str, Any]]:
    """Active tables in zone order, then top-left to bottom-right."""
    p = plan(conn, seller_id)
    order = {z: k for k, z in enumerate(p["zones"])}
    out = [i for i in p["items"] if i["kind"] == "table"]
    return sorted(out, key=lambda i: (order.get(i["zone"], 99), i["y"], i["x"]))


def table(conn, seller_id: str, no: int) -> Optional[dict[str, Any]]:
    ensure(conn, seller_id)
    return row(conn, "SELECT item_no AS no, kind, zone, name, seats, shape FROM pos_floor "
               "WHERE seller_id = ? AND item_no = ? AND kind = 'table' AND active = 1", (seller_id, no))


def name_of(conn, seller_id: str, no: int) -> Optional[str]:
    r = row(conn, "SELECT name FROM pos_floor WHERE seller_id = ? AND item_no = ?", (seller_id, no))
    return r["name"] if r else None


# --- saving --------------------------------------------------------------------------

def _int(v: Any, what: str) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        raise LayoutError("%s 값이 숫자가 아니에요." % what) from None


def _validate(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise LayoutError("배치 정보를 읽지 못했어요.")
    zones = []
    for z in payload.get("zones") or []:
        z = str(z).strip()[:10]
        if not z:
            raise LayoutError("구역 이름이 비어 있어요.")
        if z in zones:
            raise LayoutError("구역 '%s'가 두 번 있어요." % z)
        zones.append(z)
    if not zones:
        raise LayoutError("구역이 하나는 있어야 해요.")
    if len(zones) > MAX_ZONES:
        raise LayoutError("구역은 %d개까지 만들 수 있어요." % MAX_ZONES)
    raw = payload.get("items") or []
    if not isinstance(raw, list) or len(raw) > MAX_ITEMS:
        raise LayoutError("배치할 수 있는 것은 %d개까지예요." % MAX_ITEMS)
    items, names = [], {}
    for it in raw:
        if not isinstance(it, dict):
            raise LayoutError("배치 정보를 읽지 못했어요.")
        kind = it.get("kind")
        shape = str(it.get("shape") or "")
        if kind == "table":
            if shape not in TABLE_SHAPES:
                raise LayoutError("테이블 모양을 확인해 주세요.")
            name = str(it.get("name") or "").strip()[:8]
            if not name:
                raise LayoutError("이름(번호)이 없는 테이블이 있어요.")
            key = name.lower()
            if key in names:
                raise LayoutError("테이블 이름 '%s'가 두 번 있어요. 직원이 부르는 이름이라 겹치면 안 돼요." % name)
            names[key] = True
            seats = _int(it.get("seats"), "좌석 수")
            if not 1 <= seats <= 20:
                raise LayoutError("테이블 '%s'의 좌석 수는 1~20명이어야 해요." % name)
        elif kind == "fixture":
            if shape not in FIXTURES:
                raise LayoutError("시설 종류를 확인해 주세요.")
            name = str(it.get("name") or "").strip()[:10] or FIXTURES[shape]
            seats = 0
        else:
            raise LayoutError("테이블도 시설도 아닌 항목이 있어요.")
        zone = str(it.get("zone") or "").strip()[:10]
        if zone not in zones:
            raise LayoutError("'%s'가 없는 구역에 있어요." % name)
        x, y, w, h = (_int(it.get(k), k) for k in ("x", "y", "w", "h"))
        if w < 1 or h < 1 or x < 0 or y < 0 or x + w > COLS or y + h > ROWS:
            raise LayoutError("'%s'가 도면 밖으로 나갔어요." % name)
        no = it.get("no")
        items.append({"no": _int(no, "번호") if no not in (None, "") else None, "kind": kind, "zone": zone, "name": name,
                      "seats": seats, "shape": shape, "x": x, "y": y, "w": w, "h": h})
    for k, a in enumerate(items):  # nothing may sit on top of something else
        for b in items[k + 1:]:
            if a["zone"] == b["zone"] and a["x"] < b["x"] + b["w"] and b["x"] < a["x"] + a["w"] \
                    and a["y"] < b["y"] + b["h"] and b["y"] < a["y"] + a["h"]:
                raise LayoutError("'%s'와 '%s'가 %s에서 겹쳐 있어요." % (a["name"], b["name"], a["zone"]))
    return {"zones": zones, "items": items}


def save(conn, seller_id: str, payload: Any) -> dict[str, Any]:
    """Replace the store's plan with what the editor sent (see the module doc for what is kept)."""
    ensure(conn, seller_id)
    p = _validate(payload)
    existing = {r["item_no"]: r for r in rows(conn, "SELECT * FROM pos_floor WHERE seller_id = ?", (seller_id,))}
    busy = {r["table_no"] for r in rows(conn, "SELECT table_no FROM pos_tickets WHERE seller_id = ? AND status = 'open' "
                                             "AND table_no IS NOT NULL", (seller_id,))}
    kept = {i["no"] for i in p["items"] if i["kind"] == "table" and i["no"] in existing and existing[i["no"]]["kind"] == "table"}
    for no in busy - kept:
        if no in existing and existing[no]["active"]:
            raise LayoutError("테이블 '%s'에 아직 결제하지 않은 주문이 있어서 뺄 수 없어요. 결제하거나 다른 테이블로 옮긴 뒤 저장해 주세요."
                              % existing[no]["name"])
    _write(conn, seller_id, p, existing)
    return plan(conn, seller_id)


def _write(conn, seller_id: str, p: dict[str, Any], existing: dict[int, dict[str, Any]]) -> None:
    next_no = max(existing, default=0) + 1
    seen = set()
    with conn:
        for it in p["items"]:
            no = it["no"]
            if no is None or no not in existing or existing[no]["kind"] != it["kind"] or no in seen:
                no, next_no = next_no, next_no + 1
            seen.add(no)
            conn.execute("INSERT INTO pos_floor (seller_id, item_no, kind, zone, name, seats, shape, x, y, w, h, active) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1) ON CONFLICT (seller_id, item_no) DO UPDATE SET "
                         "zone = excluded.zone, name = excluded.name, seats = excluded.seats, shape = excluded.shape, "
                         "x = excluded.x, y = excluded.y, w = excluded.w, h = excluded.h, active = 1",
                         (seller_id, no, it["kind"], it["zone"], it["name"], it["seats"], it["shape"], it["x"], it["y"], it["w"], it["h"]))
            if it["kind"] == "table":  # an open check follows a renamed table; paid ones keep the old name
                conn.execute("UPDATE pos_tickets SET table_name = ? WHERE seller_id = ? AND table_no = ? AND status = 'open'",
                             (it["name"], seller_id, no))
        for no, old in existing.items():
            if no in seen:
                continue
            if old["kind"] == "table":  # past checks still name it
                conn.execute("UPDATE pos_floor SET active = 0 WHERE seller_id = ? AND item_no = ?", (seller_id, no))
            else:
                conn.execute("DELETE FROM pos_floor WHERE seller_id = ? AND item_no = ?", (seller_id, no))
        conn.execute("INSERT OR IGNORE INTO pos_settings (seller_id) VALUES (?)", (seller_id,))
        conn.execute("UPDATE pos_settings SET zones = ?, tables = ? WHERE seller_id = ?",
                     (json.dumps(p["zones"], ensure_ascii=False), sum(1 for i in p["items"] if i["kind"] == "table"), seller_id))
