"""Store chatbot: Claude with this store's tools, or a rule-based answerer without it.

Mode (CHATBOT_MODE): "llm" uses Claude through the Anthropic SDK; "rules" never
calls out; "auto" (default) uses Claude when the SDK is installed and
ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN is set. Any LLM failure falls back to
the rules for that message, so the chat never breaks.

Data boundary (architecture.md §2 keeps customer data on the seller's server):
calling Claude sends the question and tool results to Anthropic. By default
the model only gets tools over public store data -- catalog, prices, reviews,
open group buys, store info -- and "add to cart" (an action, no data back).
The customer's own orders and personalised recommendations are offered as
tools only when CHATBOT_ALLOW_CUSTOMER_DATA=true, a decision for the team
(see the merchant_api README), not a default.

Conversations are stored as text per customer in orders.sqlite (chat_messages)
and only the last turns are sent back as plain text, so no earlier thinking
block is ever replayed or edited.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from commerce.services.merchant_api import orders_service, shop_db, shop_service, social_db, social_service

_log = logging.getLogger(__name__)
MODEL = os.environ.get("CHATBOT_MODEL", "claude-opus-5-5")
HISTORY_TURNS = 8
MAX_TOOL_ROUNDS = 5


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            seller_id TEXT NOT NULL,
            customer_id_local TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            body TEXT NOT NULL,
            mode TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chat_customer ON chat_messages(seller_id, customer_id_local, created_at);
        """
    )


def history(conn, seller_id: str, customer_id: str, limit: int = 40) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT role, body, mode, created_at FROM chat_messages WHERE seller_id = ? AND customer_id_local = ? "
                        "ORDER BY created_at DESC, rowid DESC LIMIT ?", (seller_id, customer_id, limit)).fetchall()
    return [dict(r) for r in reversed(rows)]


def clear_history(conn, seller_id: str, customer_id: str) -> None:
    with conn:
        conn.execute("DELETE FROM chat_messages WHERE seller_id = ? AND customer_id_local = ?", (seller_id, customer_id))


def _save(conn, seller_id: str, customer_id: str, role: str, body: str, mode: Optional[str]) -> None:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    with conn:
        conn.execute("INSERT INTO chat_messages (seller_id, customer_id_local, role, body, mode, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                     (seller_id, customer_id, role, body, mode, now))


def mode() -> str:
    setting = os.environ.get("CHATBOT_MODE", "auto").lower()
    if setting == "rules":
        return "rules"
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return "rules"
    if setting == "llm":
        return "llm"
    has_credentials = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    return "llm" if has_credentials else "rules"


def customer_data_allowed() -> bool:
    return os.environ.get("CHATBOT_ALLOW_CUSTOMER_DATA", "false").lower() == "true"


# --- tools (shared by the LLM and the rules) ------------------------------------

@dataclass
class StoreTools:
    conn: Any
    seller_id: str
    customer_id: Optional[str]
    runtime: Any = None
    actions: list[str] = field(default_factory=list)   # what the bot did (shown under the answer)

    def _catalog(self) -> list[dict[str, Any]]:
        return orders_service.list_catalog_for_display(self.conn, seller_id=self.seller_id)

    def _brief(self, item: dict, ratings: dict) -> dict[str, Any]:
        r = ratings.get(item["item_id_local"])
        return {"item_id": item["item_id_local"], "title": item["title_text"], "price_won": item["display_price_minor"],
                "category": shop_service.category_of(item), "rating": r[0] if r else None, "reviews": r[1] if r else 0}

    def search_products(self, query: str = "", category: str = "") -> dict[str, Any]:
        catalog, ratings = self._catalog(), shop_db.rating_summary(self.conn, self.seller_id)
        hits = shop_service.search(catalog, query, category or None)
        if not hits and query:  # any word, not every word
            words = [w for w in query.split() if w]
            hits = [i for i in catalog if i["listing_status"] == "active" and any(w in i["title_text"] for w in words)]
        return {"count": len(hits), "products": [self._brief(i, ratings) for i in hits[:8]]}

    def get_product(self, item_id: str) -> dict[str, Any]:
        item = orders_service.get_catalog_item_for_display(self.conn, seller_id=self.seller_id, item_id_local=item_id)
        if item is None:
            return {"error": "no such product"}
        ratings = shop_db.rating_summary(self.conn, self.seller_id)
        reviews = shop_db.reviews_for_item(self.conn, self.seller_id, item_id)[:3]
        out = self._brief(item, ratings)
        out.update(description=item.get("description_text"), on_sale=item["listing_status"] == "active",
                   recent_reviews=[{"rating": r["rating"], "text": r["body"]} for r in reviews if r["body"]])
        if out["on_sale"]:
            out["group_buy_price_won"] = social_service.group_price(self.conn, seller_id=self.seller_id, item_id_local=item_id)
        return out

    def list_categories(self) -> dict[str, Any]:
        return {"categories": shop_service.categories(self._catalog())}

    def open_group_buys(self) -> dict[str, Any]:
        titles = {i["item_id_local"]: i["title_text"] for i in self._catalog()}
        open_ = [g for g in social_service.list_group_buys(self.conn, seller_id=self.seller_id) if g["status"] == "open"]
        return {"group_buys": [{"title": titles.get(g["item_id_local"], g["item_id_local"]), "price_won": g["unit_price_minor"],
                                "joined": g["joined_quantity"], "target": g["target_quantity"],
                                "deadline": g["deadline_at"][:10]} for g in open_]}

    def store_info(self) -> dict[str, Any]:
        settings = social_db.get_store_settings(self.conn, self.seller_id)
        return {"store": self.seller_id, "intro": settings.get("store_intro"),
                "group_buy_discount_pct": settings["group_discount_pct"],
                "group_buy_min_target": settings["group_min_target"],
                "how_to_order": "상품 화면에서 바로 주문하거나 장바구니에 담아 한 번에 주문합니다. 판매자가 수락·완료 처리합니다.",
                "cancel": "판매자가 수락하기 전(접수 상태)에는 주문내역에서 직접 취소할 수 있습니다.",
                "contact": "쪽지 메뉴로 판매자에게 직접 문의할 수 있습니다."}

    def add_to_cart(self, item_id: str, quantity: int = 1) -> dict[str, Any]:
        if not self.customer_id:
            return {"error": "로그인이 필요합니다"}
        try:
            orders_service.add_to_cart(self.conn, seller_id=self.seller_id, customer_id_local=self.customer_id,
                                       item_id_local=item_id, quantity=max(1, min(int(quantity), 20)))
        except Exception as exc:  # unknown or inactive item, bad quantity
            return {"error": "담지 못했습니다: %s" % getattr(exc, "code", exc)}
        self.actions.append("장바구니에 %s %d개를 담았어요" % (item_id, quantity))
        return {"ok": True, "cart": "/buyer/%s/cart" % self.seller_id}

    def my_recent_orders(self) -> dict[str, Any]:
        if not self.customer_id:
            return {"error": "로그인이 필요합니다"}
        titles = {i["item_id_local"]: i["title_text"] for i in self._catalog()}
        labels = {"requested": "접수", "accepted": "처리중", "completed": "완료", "cancelled": "취소"}
        orders = orders_service.list_orders_by_customer(self.conn, seller_id=self.seller_id, customer_id_local=self.customer_id)
        orders.sort(key=lambda o: o["created_at"], reverse=True)
        return {"orders": [{"date": o["created_at"][:10], "status": labels[o["status"]],
                            "items": [f"{titles.get(i['item_id_local'], i['item_id_local'])} x{i['quantity']}" for i in o["items"]]}
                           for o in orders[:5]]}

    def recommend_for_me(self) -> dict[str, Any]:
        rec = orders_service.get_recommendations_for_display(self.conn, seller_id=self.seller_id,
                                                             customer_id_local=self.customer_id or "guest",
                                                             top_n=5, runtime=self.runtime)
        catalog = {i["item_id_local"]: i for i in self._catalog()}
        ratings = shop_db.rating_summary(self.conn, self.seller_id)
        return {"source": "model" if orders_service.recommendation_label(rec) is None else "popular",
                "products": [self._brief(catalog[e["item_id_local"]], ratings) for e in rec["items"] if e["item_id_local"] in catalog]}


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def tool_definitions(customer_data: bool) -> list[dict[str, Any]]:
    tools = [
        {"name": "search_products", "description": "이 매장의 판매 중인 상품을 검색한다. query는 상품명·설명 단어, category는 카테고리 이름(선택). 빈 query는 전체.",
         "input_schema": _schema({"query": {"type": "string"}, "category": {"type": "string"}}, ["query", "category"])},
        {"name": "get_product", "description": "상품 하나의 상세: 가격, 설명, 판매 여부, 최근 리뷰, 공동구매가.",
         "input_schema": _schema({"item_id": {"type": "string"}}, ["item_id"])},
        {"name": "list_categories", "description": "이 매장의 상품 카테고리 목록.", "input_schema": _schema({}, [])},
        {"name": "open_group_buys", "description": "지금 모집 중인 공동구매 목록(구매자가 제안한 것).", "input_schema": _schema({}, [])},
        {"name": "store_info", "description": "매장 소개, 공동구매 할인율, 주문·취소·문의 방법.", "input_schema": _schema({}, [])},
        {"name": "add_to_cart", "description": "고객이 분명히 담아 달라고 할 때만 상품을 장바구니에 담는다. 결제나 주문은 하지 않는다.",
         "input_schema": _schema({"item_id": {"type": "string"}, "quantity": {"type": "integer", "description": "1~20"}},
                                 ["item_id", "quantity"])},
    ]
    if customer_data:
        tools += [
            {"name": "my_recent_orders", "description": "로그인한 고객 본인의 최근 주문 5건과 상태.", "input_schema": _schema({}, [])},
            {"name": "recommend_for_me", "description": "이 고객에게 맞춘 추천 상품(매장 추천 모델).", "input_schema": _schema({}, [])},
        ]
    for t in tools:
        t["strict"] = True
    return tools


SYSTEM = """너는 직거래 장터 '{brand}'에 입점한 매장 '{store}'의 상담 챗봇이다. 한국어로, 2~5문장으로 친절하고 간결하게 답한다.
- 상품·가격·재고·리뷰·공동구매는 반드시 도구로 확인한 내용만 말한다. 도구 결과에 없는 가격이나 사실은 지어내지 않는다.
- 상품을 말할 때는 이름과 가격(원)을 함께 쓴다.
- 장바구니 담기는 고객이 분명히 원할 때만 한다. 결제와 주문 확정은 할 수 없으니, 장바구니에서 주문하라고 안내한다.
- 배송·환불처럼 도구로 알 수 없는 것은 쪽지로 판매자에게 문의하라고 안내한다.
- 이 매장과 무관한 요청은 정중히 거절하고 매장 이야기로 돌아온다."""


def _llm_reply(tools_impl: StoreTools, question: str, past: list[dict[str, Any]], brand: str) -> str:
    import anthropic

    client = anthropic.Anthropic(timeout=60.0)
    customer_data = customer_data_allowed()
    tools = tool_definitions(customer_data)
    allowed = {t["name"] for t in tools}
    messages: list[dict[str, Any]] = [{"role": m["role"], "content": m["body"]} for m in past[-2 * HISTORY_TURNS:]]
    if messages and messages[0]["role"] != "user":
        messages = messages[1:]
    messages.append({"role": "user", "content": question})
    system = SYSTEM.format(brand=brand, store=tools_impl.seller_id) + (
        "" if tools_impl.customer_id else "\n- 고객이 로그인하지 않았다. 장바구니 담기는 로그인 뒤에 할 수 있다고 안내한다.")
    for _ in range(MAX_TOOL_ROUNDS):
        response = client.beta.messages.create(
            model=MODEL, max_tokens=4000, system=system, tools=tools, messages=messages,
            output_config={"effort": "low"},  # short store Q&A
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return "죄송해요, 그 요청은 도와드리기 어려워요. 상품이나 주문에 대해 물어봐 주세요."
        calls = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason != "tool_use" or not calls:
            text = "".join(b.text for b in response.content if b.type == "text").strip()
            return text or "무엇을 도와드릴까요?"
        messages.append({"role": "assistant", "content": response.content})  # unchanged, thinking included
        results = []
        for call in calls:
            if call.name not in allowed:
                payload, is_error = {"error": "unknown tool"}, True
            else:
                try:
                    payload, is_error = getattr(tools_impl, call.name)(**call.input), False
                except TypeError:  # arguments that do not match the schema
                    payload, is_error = {"error": "invalid arguments"}, True
            results.append({"type": "tool_result", "tool_use_id": call.id, "is_error": is_error,
                            "content": json.dumps(payload, ensure_ascii=False)})
        messages.append({"role": "user", "content": results})
    return "확인할 내용이 많아 정리하지 못했어요. 조금 더 구체적으로 물어봐 주세요."


# --- rule-based answerer ------------------------------------------------------------

def _won(n: int) -> str:
    return format(n, ",") + "원"


def _rules_reply(tools_impl: StoreTools, question: str) -> str:
    q = question.strip()
    if re.search(r"(담아|넣어|추가해)", q):
        found = tools_impl.search_products(re.sub(r"(장바구니|에|좀|담아|넣어|추가해|줘|주세요|요)", " ", q).strip())["products"]
        if not found:
            return "어떤 상품을 담을까요? 상품 이름을 알려 주세요."
        result = tools_impl.add_to_cart(found[0]["item_id"], 1)
        if "error" in result:
            return "장바구니에 담으려면 로그인해 주세요." if "로그인" in result["error"] else result["error"]
        return "%s(%s) 1개를 장바구니에 담았어요. 장바구니에서 한 번에 주문할 수 있어요." % (found[0]["title"], _won(found[0]["price_won"]))
    if re.search(r"(주문|배송|언제 와|취소)", q):
        if not tools_impl.customer_id:
            return "주문 확인은 로그인 후 '주문내역'에서 할 수 있어요. 접수 상태의 주문은 거기서 바로 취소할 수 있어요."
        orders = tools_impl.my_recent_orders()["orders"]
        if not orders:
            return "아직 주문이 없어요."
        o = orders[0]
        return "가장 최근 주문(%s)은 '%s' 상태예요: %s. 자세한 건 주문내역에서 볼 수 있어요." % (o["date"], o["status"], ", ".join(o["items"]))
    if re.search(r"(공동구매|공구|같이 사)", q):
        gbs = tools_impl.open_group_buys()["group_buys"]
        disc = tools_impl.store_info()["group_buy_discount_pct"]
        if not gbs:
            return "지금 모집 중인 공동구매는 없어요. 공동구매 메뉴에서 직접 제안하면 %d%% 할인가로 열 수 있어요." % disc
        top = gbs[0]
        return "모집 중인 공동구매가 %d건 있어요. 예: %s %s (%d/%d개). 공동구매 메뉴에서 참여할 수 있어요." % (
            len(gbs), top["title"], _won(top["price_won"]), top["joined"], top["target"])
    if re.search(r"(추천|뭐가 좋|인기|베스트)", q):
        rec = tools_impl.recommend_for_me()["products"][:3]
        if not rec:
            return "아직 추천할 상품이 없어요."
        return "이 상품들을 추천해요: " + ", ".join("%s(%s)" % (p["title"], _won(p["price_won"])) for p in rec) + "."
    if re.search(r"(문의|환불|교환|픽업|영업)", q):
        return tools_impl.store_info()["contact"] + " 공동구매·장바구니·주문 관련 질문은 저에게 해 주세요."
    words = re.sub(r"(있어요|있나요|있어|얼마|가격|찾아|줘|주세요|요|\?)", " ", q).strip()
    found = tools_impl.search_products(words)["products"] if words else []
    if found:
        p = found[0]
        extra = " 외 %d개" % (len(found) - 1) if len(found) > 1 else ""
        rating = " · ★%s" % p["rating"] if p["rating"] else ""
        return "%s는 %s이에요%s.%s 상품 화면에서 바로 주문하거나 '담아 줘'라고 말해 주세요." % (p["title"], _won(p["price_won"]), rating, extra)
    return "상품 찾기('감귤 있어요?'), 추천('뭐가 좋아요?'), 장바구니 담기('우유 담아 줘'), 주문 확인, 공동구매를 도와드릴 수 있어요."


@dataclass(frozen=True)
class Reply:
    text: str
    mode: str          # "llm" | "rules" | "rules-fallback"
    actions: tuple[str, ...]


def ask(conn, *, seller_id: str, customer_id: Optional[str], question: str, brand: str, runtime=None,
        llm: Optional[Callable[..., str]] = None) -> Reply:
    """Answer one question and store both sides. `llm` replaces the Claude call in tests."""
    question = (question or "").strip()[:500]
    if not question:
        return Reply("무엇을 도와드릴까요?", "rules", ())
    who = customer_id or "guest"
    past = history(conn, seller_id, who) if customer_id else []
    tools_impl = StoreTools(conn, seller_id, customer_id, runtime)
    used = mode() if llm is None else "llm"
    text = None
    if used == "llm":
        try:
            text = (llm or _llm_reply)(tools_impl, question, past, brand)
        except Exception as exc:  # network, auth, rate limit: answer anyway
            _log.warning("chatbot LLM call failed, using rules: %s", exc)
            used = "rules-fallback"
    if text is None:
        text = _rules_reply(tools_impl, question)
        used = used if used == "rules-fallback" else "rules"
    if customer_id:
        _save(conn, seller_id, who, "user", question, None)
        _save(conn, seller_id, who, "assistant", text, used)
    return Reply(text, used, tuple(tools_impl.actions))
