"""Shop extras: wishlist, reviews, buyer order cancel, seller product editing and review replies."""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import RedirectResponse

from commerce.packages.contracts.errors import ContractError
from commerce.services.merchant_api import orders_service, shop_db, shop_service
from commerce.services.merchant_api.routes_sns import ScreenDeps


def register(app: FastAPI, d: ScreenDeps) -> None:
    # --- buyer -----------------------------------------------------------------

    @app.post("/buyer/{seller_id}/wishlist/{item_id_local}")
    def buyer_toggle_wish(seller_id: str, item_id_local: str, conn=Depends(d.get_conn),
                          customer=Depends(d.require_customer_form), back: str = Form("item")):
        shop_service.toggle_wish(conn, seller_id=seller_id, customer_id_local=customer["customer_id_local"],
                                 item_id_local=item_id_local)
        target = {"wishlist": f"/buyer/{seller_id}/wishlist", "home": f"/buyer/{seller_id}/"}.get(
            back, f"/buyer/{seller_id}/items/{item_id_local}")
        return RedirectResponse(target, status_code=303)

    @app.get("/buyer/{seller_id}/wishlist")
    def buyer_wishlist(seller_id: str, request: Request, conn=Depends(d.get_conn), customer=Depends(d.require_customer)):
        catalog = {i["item_id_local"]: i for i in orders_service.list_catalog_for_display(conn, seller_id=seller_id)}
        items = [catalog[i] for i in shop_db.wishlist(conn, seller_id, customer["customer_id_local"]) if i in catalog]
        return d.buyer_templates.TemplateResponse(request, "wishlist.html", {
            "seller_id": seller_id, "active_tab": "wishlist", "customer": customer, "items": items,
            "ratings": shop_db.rating_summary(conn, seller_id),
        })

    @app.post("/buyer/{seller_id}/items/{item_id_local}/reviews")
    def buyer_review(seller_id: str, item_id_local: str, conn=Depends(d.get_conn), customer=Depends(d.require_customer_form),
                     rating: int = Form(...), body: str = Form("")):
        try:
            shop_service.write_review(conn, seller_id=seller_id, customer_id_local=customer["customer_id_local"],
                                      item_id_local=item_id_local, rating=rating, body=body)
        except ContractError:
            return RedirectResponse(f"/buyer/{seller_id}/items/{item_id_local}?review=denied#reviews", status_code=303)
        return RedirectResponse(f"/buyer/{seller_id}/items/{item_id_local}#reviews", status_code=303)

    @app.post("/buyer/{seller_id}/orders/{order_id}/cancel")
    def buyer_cancel(seller_id: str, order_id: str, request: Request, conn=Depends(d.get_conn),
                     customer=Depends(d.require_customer_form)):
        try:
            shop_service.cancel_by_buyer(conn, seller_id=seller_id, customer_id_local=customer["customer_id_local"],
                                         order_id=order_id, runtime=request.app.state.merchant.runtime)
        except ContractError as exc:
            return RedirectResponse(f"/buyer/{seller_id}/orders?e={exc.code}", status_code=303)
        return RedirectResponse(f"/buyer/{seller_id}/orders?ok=cancelled", status_code=303)

    # --- seller ----------------------------------------------------------------

    @app.get("/seller/{seller_id}/products/{item_id_local}")
    def seller_product(seller_id: str, item_id_local: str, request: Request, conn=Depends(d.get_conn),
                       staff=Depends(d.require_seller)):
        item = orders_service.get_catalog_item_for_display(conn, seller_id=seller_id, item_id_local=item_id_local)
        if item is None:
            raise ContractError("NOT_FOUND", "/item_id_local")
        return d.seller_templates.TemplateResponse(request, "product_edit.html", {
            "seller_id": seller_id, "active_tab": "products", "staff": staff, "item": item,
            "reviews": shop_db.reviews_for_item(conn, seller_id, item_id_local),
            "rating": shop_db.rating_summary(conn, seller_id).get(item_id_local),
            "wishes": shop_db.wish_counts(conn, seller_id).get(item_id_local, 0),
        })

    @app.post("/seller/{seller_id}/products/{item_id_local}")
    def seller_edit_product(seller_id: str, item_id_local: str, request: Request, conn=Depends(d.get_conn),
                            staff=Depends(d.require_seller_form), title_text: str = Form(...),
                            display_price_minor: int = Form(...), category: str = Form(""),
                            description_text: str = Form(""), listing_status: str = Form("active")):
        shop_service.edit_product(conn, seller_id=seller_id, item_id_local=item_id_local, title_text=title_text,
                                  display_price_minor=display_price_minor,
                                  category_path=[c.strip() for c in category.split(">") if c.strip()],
                                  description_text=description_text, listing_status=listing_status,
                                  runtime=request.app.state.merchant.runtime)
        return RedirectResponse(f"/seller/{seller_id}/products/{item_id_local}?saved=1", status_code=303)

    @app.post("/seller/{seller_id}/products/{item_id_local}/reviews/{customer_id_local}/reply")
    def seller_reply_review(seller_id: str, item_id_local: str, customer_id_local: str, conn=Depends(d.get_conn),
                            staff=Depends(d.require_seller_form), reply: str = Form("")):
        shop_service.reply_review(conn, seller_id=seller_id, item_id_local=item_id_local,
                                  customer_id_local=customer_id_local, reply=reply)
        return RedirectResponse(f"/seller/{seller_id}/products/{item_id_local}#reviews", status_code=303)
