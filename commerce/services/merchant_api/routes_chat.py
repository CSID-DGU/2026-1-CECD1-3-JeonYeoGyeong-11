"""Chatbot screen and endpoint (see chatbot.py for modes and the data boundary)."""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse

from commerce.packages.contracts.errors import ContractError
from commerce.services.merchant_api import chatbot, session
from commerce.services.merchant_api.routes_sns import ScreenDeps

SUGGESTIONS = ("오늘 뭐가 좋아요?", "감귤 있어요?", "모집 중인 공동구매 있어요?", "주문 취소는 어떻게 해요?")


def register(app: FastAPI, d: ScreenDeps, brand: str) -> None:
    @app.get("/buyer/{seller_id}/chat")
    def buyer_chat(seller_id: str, request: Request, conn=Depends(d.get_conn), customer=Depends(d.current_customer)):
        past = chatbot.history(conn, seller_id, customer["customer_id_local"]) if customer else []
        return d.buyer_templates.TemplateResponse(request, "chat.html", {
            "seller_id": seller_id, "active_tab": "chat", "customer": customer, "messages": past,
            "mode": chatbot.mode(), "customer_data": chatbot.customer_data_allowed(), "suggestions": SUGGESTIONS,
        })

    @app.post("/buyer/{seller_id}/chat")
    def buyer_ask(seller_id: str, request: Request, conn=Depends(d.get_conn), customer=Depends(d.current_customer),
                  question: str = Form(...), csrf_token: Optional[str] = Form(None)):
        # Logged-in chats are stored and can change the cart, so they need the session's CSRF token.
        # A guest has no session to forge; their questions are answered and not stored.
        if customer and not session.check_csrf(request.cookies.get(session.customer_cookie(seller_id)), csrf_token):
            raise ContractError("FORBIDDEN", "/csrf_token")
        reply = chatbot.ask(conn, seller_id=seller_id, customer_id=customer["customer_id_local"] if customer else None,
                            question=question, brand=brand, runtime=request.app.state.merchant.runtime)
        if "application/json" in request.headers.get("accept", ""):
            return JSONResponse({"reply": reply.text, "mode": reply.mode, "actions": list(reply.actions)})
        if customer:
            return RedirectResponse(f"/buyer/{seller_id}/chat#end", status_code=303)
        return d.buyer_templates.TemplateResponse(request, "chat.html", {
            "seller_id": seller_id, "active_tab": "chat", "customer": None, "suggestions": SUGGESTIONS,
            "mode": chatbot.mode(), "customer_data": chatbot.customer_data_allowed(),
            "messages": [{"role": "user", "body": question}, {"role": "assistant", "body": reply.text, "mode": reply.mode}],
        })

    @app.post("/buyer/{seller_id}/chat/clear")
    def buyer_chat_clear(seller_id: str, conn=Depends(d.get_conn), customer=Depends(d.require_customer_form)):
        chatbot.clear_history(conn, seller_id, customer["customer_id_local"])
        return RedirectResponse(f"/buyer/{seller_id}/chat", status_code=303)
