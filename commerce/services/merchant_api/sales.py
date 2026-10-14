"""The seller's sales numbers for the overview: daily revenue, top products, repeat buyers.

Counted from this store's orders.sqlite only (completed orders, by completion day,
KST). Revenue is items x unit price as ordered; shipping fees are not revenue here.
"""
from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict
from typing import Any, Optional

KST = dt.timezone(dt.timedelta(hours=9))


def _day(ts: Optional[str]) -> Optional[dt.date]:
    if not ts:
        return None
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(KST).date()


def _total(order: dict[str, Any]) -> int:
    return sum(i["quantity"] * i["unit_price_minor"] for i in order["items"])


def summary(orders: list[dict[str, Any]], titles: dict[str, str], *, days: int = 14,
            today: Optional[dt.date] = None) -> dict[str, Any]:
    done = [o for o in orders if o["status"] == "completed" and o.get("completed_at")]
    latest = max((_day(o["completed_at"]) for o in done), default=None)
    today = today or max(filter(None, [dt.datetime.now(KST).date(), latest]))
    by_day: dict[dt.date, int] = defaultdict(int)
    count_by_day: dict[dt.date, int] = defaultdict(int)
    for o in done:
        d = _day(o["completed_at"])
        by_day[d] += _total(o)
        count_by_day[d] += 1
    series = []
    for n in range(days - 1, -1, -1):
        d = today - dt.timedelta(days=n)
        series.append({"date": d.isoformat(), "label": "%d/%d" % (d.month, d.day), "revenue": by_day.get(d, 0),
                       "orders": count_by_day.get(d, 0)})

    def window(n_days: int) -> list[dict[str, Any]]:
        start = today - dt.timedelta(days=n_days - 1)
        return [o for o in done if start <= _day(o["completed_at"]) <= today]

    last7, prev7 = window(7), [o for o in window(14) if _day(o["completed_at"]) < today - dt.timedelta(days=6)]
    last30 = window(30)
    rev7, rev_prev7 = sum(map(_total, last7)), sum(map(_total, prev7))
    item_rev: Counter = Counter()
    item_qty: Counter = Counter()
    for o in last30:
        for i in o["items"]:
            item_rev[i["item_id_local"]] += i["quantity"] * i["unit_price_minor"]
            item_qty[i["item_id_local"]] += i["quantity"]
    buyers = Counter(o["customer_id_local"] for o in done)
    repeat = sum(1 for n in buyers.values() if n >= 2)
    top = [{"item_id_local": item, "title": titles.get(item, item), "revenue": rev, "quantity": item_qty[item]}
           for item, rev in item_rev.most_common(5)]
    peak = max((p["revenue"] for p in series), default=0)
    return {
        "series": series, "peak": peak, "today": today.isoformat(),
        "revenue_today": by_day.get(today, 0), "orders_today": count_by_day.get(today, 0),
        "revenue_7d": rev7, "revenue_prev_7d": rev_prev7,
        "change_7d": round(100 * (rev7 - rev_prev7) / rev_prev7) if rev_prev7 else None,
        "revenue_30d": sum(map(_total, last30)), "orders_30d": len(last30),
        "aov_30d": round(sum(map(_total, last30)) / len(last30)) if last30 else 0,
        "buyers": len(buyers), "repeat_buyers": repeat,
        "repeat_share": round(100 * repeat / len(buyers)) if buyers else 0,
        "top_items": top, "top_peak": top[0]["revenue"] if top else 0,
    }


def bar_chart(series: list[dict[str, Any]], peak: int, *, width: int = 640, height: int = 180) -> dict[str, Any]:
    """Geometry for the inline SVG bar chart (the template draws it): one bar per day, rounded top."""
    pad_l, pad_r, pad_t, pad_b = 8, 8, 12, 26
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    step = plot_w / max(1, len(series))
    bar_w = max(4.0, min(28.0, step * 0.55))
    scale_top = _nice(peak)
    bars = []
    for n, p in enumerate(series):
        h = 0 if not scale_top else plot_h * p["revenue"] / scale_top
        x = pad_l + step * n + (step - bar_w) / 2
        base, top, r = pad_t + plot_h, pad_t + plot_h - h, min(4.0, bar_w / 2, h)
        path = ("M%.1f,%.1f V%.1f Q%.1f,%.1f %.1f,%.1f H%.1f Q%.1f,%.1f %.1f,%.1f V%.1f Z"
                % (x, base, top + r, x, top, x + r, top, x + bar_w - r, x + bar_w, top, x + bar_w, top + r, base)) if h > 0 else ""
        bars.append(dict(p, d=path, x=round(x, 1), y=round(top, 1), w=round(bar_w, 1), h=round(h, 1),
                         cx=round(x + bar_w / 2, 1), hit_x=round(pad_l + step * n, 1), hit_w=round(step, 1),
                         show_label=(n % 2 == (len(series) - 1) % 2)))
    grid = [{"y": round(pad_t + plot_h * (1 - f), 1), "value": int(scale_top * f)} for f in (0.5, 1.0)] if scale_top else []
    return {"width": width, "height": height, "bars": bars, "grid": grid, "base_y": pad_t + plot_h,
            "label_y": height - 8, "plot_left": pad_l, "plot_right": width - pad_r, "radius": min(4, bar_w / 2)}


def _nice(value: int) -> int:
    if value <= 0:
        return 0
    magnitude = 10 ** (len(str(int(value))) - 1)
    for m in (1, 2, 2.5, 5, 10):
        if value <= m * magnitude:
            return int(m * magnitude)
    return int(10 * magnitude)
