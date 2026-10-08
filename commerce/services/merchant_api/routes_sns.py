"""Screens for the store SNS: buyer explore feed and post pages, seller studio.

Registered by main.create_app with the same login / CSRF / connection
dependencies the other screens use (ScreenDeps), so the rules stay in one place.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional
from urllib.parse import quote

from markupsafe import Markup

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response

from commerce.packages.contracts.errors import ContractError
from commerce.services.merchant_api import media, orders_service, sns_service


@dataclass(frozen=True)
class ScreenDeps:
    get_conn: Callable
    current_customer: Callable
    require_customer: Callable
    require_customer_form: Callable
    require_seller: Callable
    require_seller_form: Callable
    buyer_templates: Any
    seller_templates: Any


def _read_uploads(files: list[UploadFile]) -> list[bytes]:
    out = []
    for f in files or []:
        if not f or not f.filename:
            continue
        data = f.file.read(media.MAX_UPLOAD_BYTES + 1)
        if len(data) > media.MAX_UPLOAD_BYTES:
            raise media.MediaError("파일은 %dMB까지 올릴 수 있습니다." % (media.MAX_UPLOAD_BYTES // (1024 * 1024)))
        out.append(data)
    return out


def caption_html(text: Optional[str], seller_id: str) -> Markup:
    """Escaped caption with line breaks and #hashtags linked to the feed filter."""
    escaped = html.escape(text or "")
    linked = re.sub(r"#([0-9A-Za-z_가-힣]{1,30})",
                    lambda m: '<a class="hashtag" href="/buyer/%s/feed?tag=%s">#%s</a>'
                              % (html.escape(seller_id), quote(m.group(1)), m.group(1)), escaped)
    return Markup(linked.replace("\n", "<br>"))


def register(app: FastAPI, d: ScreenDeps) -> None:
    for templates in (d.buyer_templates, d.seller_templates):
        templates.env.filters["caption"] = caption_html
    def folder(request: Request):
        return media.media_dir(request.app.state.merchant.merchant_db_path)

    # --- media ---------------------------------------------------------------

    @app.get("/media/{seller_id}/{name}")
    def serve_media(seller_id: str, name: str, request: Request):
        if seller_id != request.app.state.merchant.seller_id:
            return Response(status_code=404)
        path = media.resolve(folder(request), name)
        if path is None:
            return Response(status_code=404)
        ext = name.rsplit(".", 1)[1]
        headers = {"Cache-Control": "public, max-age=86400"}
        if ext == "svg":
            headers.update(media.SVG_HEADERS)
        return FileResponse(path, media_type=media.CONTENT_TYPES[ext], headers=headers)

    # --- buyer -----------------------------------------------------------------

    @app.get("/buyer/{seller_id}/feed")
    def buyer_feed(seller_id: str, request: Request, conn=Depends(d.get_conn), customer=Depends(d.current_customer),
                   tag: Optional[str] = None):
        customer_id = customer["customer_id_local"] if customer else None
        posts = sns_service.explore(conn, seller_id=seller_id, customer_id_local=customer_id,
                                    runtime=request.app.state.merchant.runtime)
        if tag:
            posts = [p for p in posts if tag in p["hashtags"]]
        hashtags: dict[str, int] = {}
        for p in posts:
            for h in p["hashtags"]:
                hashtags[h] = hashtags.get(h, 0) + 1
        return d.buyer_templates.TemplateResponse(request, "feed.html", {
            "seller_id": seller_id, "active_tab": "feed", "customer": customer, "posts": posts, "tag": tag,
            "top_tags": sorted(hashtags, key=lambda h: -hashtags[h])[:12],
        })

    @app.get("/buyer/{seller_id}/posts/{post_id}")
    def buyer_post(seller_id: str, post_id: str, request: Request, conn=Depends(d.get_conn),
                   customer=Depends(d.current_customer)):
        customer_id = customer["customer_id_local"] if customer else None
        post = sns_service.post_detail(conn, seller_id=seller_id, post_id=post_id, customer_id_local=customer_id)
        if post is None:
            raise ContractError("NOT_FOUND", "/post_id")
        sns_service.record_view(conn, seller_id=seller_id, post_id=post_id, viewer=customer_id or "guest")
        return d.buyer_templates.TemplateResponse(request, "post.html", {
            "seller_id": seller_id, "active_tab": "feed", "customer": customer, "post": post,
        })

    @app.post("/buyer/{seller_id}/posts/{post_id}/like")
    def buyer_like(seller_id: str, post_id: str, conn=Depends(d.get_conn), customer=Depends(d.require_customer_form),
                   back: str = Form("post")):
        sns_service.toggle_like(conn, seller_id=seller_id, post_id=post_id, customer_id_local=customer["customer_id_local"])
        target = f"/buyer/{seller_id}/feed#p-{post_id}" if back == "feed" else f"/buyer/{seller_id}/posts/{post_id}"
        return RedirectResponse(target, status_code=303)

    @app.post("/buyer/{seller_id}/posts/{post_id}/comments")
    def buyer_comment(seller_id: str, post_id: str, conn=Depends(d.get_conn), customer=Depends(d.require_customer_form),
                      body: str = Form(...)):
        sns_service.add_comment(conn, seller_id=seller_id, post_id=post_id, body=body,
                                customer_id_local=customer["customer_id_local"])
        return RedirectResponse(f"/buyer/{seller_id}/posts/{post_id}#comments", status_code=303)

    @app.post("/buyer/{seller_id}/posts/{post_id}/comments/{comment_id}/delete")
    def buyer_delete_comment(seller_id: str, post_id: str, comment_id: str, conn=Depends(d.get_conn),
                             customer=Depends(d.require_customer_form)):
        sns_service.delete_comment(conn, seller_id=seller_id, comment_id=comment_id,
                                   customer_id_local=customer["customer_id_local"])
        return RedirectResponse(f"/buyer/{seller_id}/posts/{post_id}#comments", status_code=303)

    # --- seller studio -----------------------------------------------------------

    def studio(request: Request, conn, seller_id: str, staff, error: Optional[str] = None, form: Optional[dict] = None):
        return d.seller_templates.TemplateResponse(request, "feed.html", {
            "seller_id": seller_id, "active_tab": "feed", "staff": staff, "error": error, "form": form or {},
            "posts": sns_service.insights(conn, seller_id=seller_id),
            "catalog": [i for i in orders_service.list_catalog_for_display(conn, seller_id=seller_id)
                        if i["listing_status"] == "active"],
            "max_files": media.MAX_FILES_PER_POST,
        }, status_code=400 if error else 200)

    @app.get("/seller/{seller_id}/feed")
    def seller_feed(seller_id: str, request: Request, conn=Depends(d.get_conn), staff=Depends(d.require_seller)):
        return studio(request, conn, seller_id, staff)

    @app.post("/seller/{seller_id}/feed")
    def seller_create_post(seller_id: str, request: Request, conn=Depends(d.get_conn),
                           staff=Depends(d.require_seller_form), kind: str = Form("article"), caption: str = Form(""),
                           item_ids: list[str] = Form(default=[]), files: list[UploadFile] = File(default=[])):
        try:
            sns_service.create_post(conn, folder(request), seller_id=seller_id, kind=kind, caption=caption,
                                    item_ids=item_ids, uploads=_read_uploads(files))
        except media.MediaError as exc:
            return studio(request, conn, seller_id, staff, str(exc), {"kind": kind, "caption": caption, "items": item_ids})
        except ContractError as exc:
            message = {"MISSING_REQUIRED_FIELD": "내용을 입력해 주세요.", "SCHEMA_INVALID": "내용이 너무 길거나 태그·파일이 너무 많습니다.",
                       "NOT_FOUND": "판매 중인 상품만 태그할 수 있습니다."}.get(exc.code, exc.code)
            return studio(request, conn, seller_id, staff, message, {"kind": kind, "caption": caption, "items": item_ids})
        return RedirectResponse(f"/seller/{seller_id}/feed", status_code=303)

    @app.get("/seller/{seller_id}/posts/{post_id}")
    def seller_post(seller_id: str, post_id: str, request: Request, conn=Depends(d.get_conn),
                    staff=Depends(d.require_seller)):
        post = sns_service.post_detail(conn, seller_id=seller_id, post_id=post_id, customer_id_local=None)
        if post is None:
            raise ContractError("NOT_FOUND", "/post_id")
        return d.seller_templates.TemplateResponse(request, "post_edit.html", {
            "seller_id": seller_id, "active_tab": "feed", "staff": staff, "post": post,
            "catalog": [i for i in orders_service.list_catalog_for_display(conn, seller_id=seller_id)
                        if i["listing_status"] == "active"],
        })

    @app.post("/seller/{seller_id}/posts/{post_id}")
    def seller_update_post(seller_id: str, post_id: str, conn=Depends(d.get_conn), staff=Depends(d.require_seller_form),
                           caption: str = Form(...), item_ids: list[str] = Form(default=[])):
        sns_service.update_post(conn, seller_id=seller_id, post_id=post_id, caption=caption, item_ids=item_ids)
        return RedirectResponse(f"/seller/{seller_id}/posts/{post_id}", status_code=303)

    @app.post("/seller/{seller_id}/posts/{post_id}/delete")
    def seller_delete_post(seller_id: str, post_id: str, conn=Depends(d.get_conn), staff=Depends(d.require_seller_form)):
        sns_service.delete_post(conn, seller_id=seller_id, post_id=post_id)
        return RedirectResponse(f"/seller/{seller_id}/feed", status_code=303)

    @app.post("/seller/{seller_id}/posts/{post_id}/comments")
    def seller_reply(seller_id: str, post_id: str, conn=Depends(d.get_conn), staff=Depends(d.require_seller_form),
                     body: str = Form(...)):
        sns_service.add_comment(conn, seller_id=seller_id, post_id=post_id, body=body)
        return RedirectResponse(f"/seller/{seller_id}/posts/{post_id}#comments", status_code=303)

    @app.post("/seller/{seller_id}/posts/{post_id}/comments/{comment_id}/delete")
    def seller_delete_comment(seller_id: str, post_id: str, comment_id: str, conn=Depends(d.get_conn),
                              staff=Depends(d.require_seller_form)):
        sns_service.delete_comment(conn, seller_id=seller_id, comment_id=comment_id)
        return RedirectResponse(f"/seller/{seller_id}/posts/{post_id}#comments", status_code=303)
