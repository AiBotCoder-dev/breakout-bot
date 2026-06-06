"""
broker.py — Alpaca PAPER trading client (real broker, simulated money).

Replaces the bot's internal paper simulation with a real broker paper account so
you get HONEST fills + pricing and a clean track record to judge profitability.

Raw REST (no SDK dependency) to https://paper-api.alpaca.markets, reusing the
same APCA auth headers the data layer already uses.

CREDENTIALS (set as env vars / Streamlit secrets / GitHub secrets):
  ALPACA_PAPER_KEY / ALPACA_PAPER_SECRET   (preferred — your PAPER account keys)
  ... falls back to ALPACA_API_KEY / ALPACA_API_SECRET if those are paper keys.

SAFETY:
  • This NEVER touches a live/funded account — base URL is hard-coded to the
    paper endpoint. There is no live-trading path in this module.
  • Auto-trading is OFF until you set BROKER_MODE=alpaca_paper (see monitor.py).
  • Get free paper keys: alpaca.markets -> Paper Trading -> API keys. You can
    set the paper account's starting equity (e.g. $200) in the Alpaca dashboard.

SCOPE: stocks (the validated momentum edge). Options via Alpaca paper are
possible but need options enabled on the account + OCC symbols — a clean
follow-on once the stock track record is trusted.
"""

from __future__ import annotations

import os

try:
    import requests
except Exception:                       # pragma: no cover
    requests = None

PAPER_BASE = "https://paper-api.alpaca.markets"


def _creds() -> tuple[str, str]:
    key = (os.environ.get("ALPACA_PAPER_KEY", "")
           or os.environ.get("ALPACA_API_KEY", "")).strip()
    secret = (os.environ.get("ALPACA_PAPER_SECRET", "")
              or os.environ.get("ALPACA_API_SECRET", "")).strip()
    return key, secret


class AlpacaPaperBroker:
    def __init__(self):
        self.base = PAPER_BASE
        self.key, self.secret = _creds()

    # ── infra ─────────────────────────────────────────────────────────────────
    def available(self) -> bool:
        return bool(self.key and self.secret and requests is not None)

    def _headers(self) -> dict:
        return {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret}

    def _get(self, path: str, params: dict | None = None):
        r = requests.get(f"{self.base}{path}", headers=self._headers(),
                         params=params or {}, timeout=15)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict):
        r = requests.post(f"{self.base}{path}", headers=self._headers(),
                          json=body, timeout=15)
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
        return r.json()

    def _delete(self, path: str):
        r = requests.delete(f"{self.base}{path}", headers=self._headers(), timeout=15)
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
        return True

    def test_connection(self) -> dict:
        """Return {ok, account_number, status, equity} or {ok: False, error}."""
        if not self.available():
            return {"ok": False, "error": "No Alpaca paper credentials set "
                    "(ALPACA_PAPER_KEY / ALPACA_PAPER_SECRET)."}
        try:
            a = self._get("/v2/account")
            return {"ok": True, "account_number": a.get("account_number"),
                    "status": a.get("status"), "equity": float(a.get("equity", 0))}
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}

    # ── account / positions / orders ───────────────────────────────────────────
    def get_account(self) -> dict:
        try:
            a = self._get("/v2/account")
            eq = float(a.get("equity", 0)); last_eq = float(a.get("last_equity", 0) or eq)
            return {
                "equity": eq, "cash": float(a.get("cash", 0)),
                "buying_power": float(a.get("buying_power", 0)),
                "last_equity": last_eq,
                "day_pnl": round(eq - last_eq, 2),
                "day_pnl_pct": round((eq / last_eq - 1) * 100, 2) if last_eq else 0,
                "status": a.get("status"),
                "pattern_day_trader": a.get("pattern_day_trader", False),
            }
        except Exception as e:
            return {"error": str(e)[:200]}

    def get_positions(self) -> list:
        try:
            rows = self._get("/v2/positions")
        except Exception:
            return []
        out = []
        for p in rows:
            try:
                out.append({
                    "ticker": p.get("symbol"),
                    "qty": float(p.get("qty", 0)),
                    "avg_entry": float(p.get("avg_entry_price", 0)),
                    "current": float(p.get("current_price", 0) or 0),
                    "market_value": float(p.get("market_value", 0) or 0),
                    "cost_basis": float(p.get("cost_basis", 0) or 0),
                    "unrealized_pnl": float(p.get("unrealized_pl", 0) or 0),
                    "unrealized_pct": round(float(p.get("unrealized_plpc", 0) or 0) * 100, 2),
                    "side": p.get("side", "long"),
                })
            except Exception:
                continue
        return out

    def get_orders(self, status: str = "all", limit: int = 50) -> list:
        try:
            rows = self._get("/v2/orders", {"status": status, "limit": limit,
                                            "direction": "desc"})
        except Exception:
            return []
        out = []
        for o in rows:
            out.append({
                "ticker": o.get("symbol"), "side": o.get("side"),
                "qty": o.get("qty"), "type": o.get("type"),
                "order_class": o.get("order_class"),
                "status": o.get("status"),
                "filled_qty": o.get("filled_qty"),
                "filled_avg_price": o.get("filled_avg_price"),
                "submitted_at": (o.get("submitted_at") or "")[:19],
                "limit_price": o.get("limit_price"),
                "stop_price": o.get("stop_price"),
            })
        return out

    # ── trading ────────────────────────────────────────────────────────────────
    def get_price(self, ticker: str) -> float | None:
        try:
            from data_providers import get_live_prices
            p = get_live_prices((ticker.upper(),))
            v = p.get(ticker.upper())
            return float(v) if v else None
        except Exception:
            return None

    def submit_bracket_order(self, ticker: str, position_dollars: float,
                             stop: float, target: float,
                             price: float | None = None) -> dict:
        """
        Market buy sized to ~position_dollars (whole shares) with an attached
        bracket: take-profit limit at `target`, stop-loss at `stop`.
        Returns {ok, qty, order_id} or {ok: False, error}.
        """
        if not self.available():
            return {"ok": False, "error": "broker not configured"}
        ticker = ticker.upper()
        px = price or self.get_price(ticker)
        if not px or px <= 0:
            return {"ok": False, "error": "no price"}
        qty = int(position_dollars // px)
        if qty < 1:
            return {"ok": False, "error": f"position ${position_dollars:.0f} < 1 "
                    f"share of {ticker} (${px:.2f}) — too expensive for this size"}
        # Alpaca requires target > entry > stop for a long bracket; sanitize.
        tp = round(max(target, px * 1.02), 2)
        sl = round(min(stop, px * 0.98), 2)
        body = {
            "symbol": ticker, "qty": str(qty), "side": "buy", "type": "market",
            "time_in_force": "gtc", "order_class": "bracket",
            "take_profit": {"limit_price": str(tp)},
            "stop_loss": {"stop_price": str(sl)},
        }
        try:
            o = self._post("/v2/orders", body)
            return {"ok": True, "qty": qty, "order_id": o.get("id"),
                    "entry_est": px, "tp": tp, "sl": sl,
                    "cost_est": round(qty * px, 2)}
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}

    def close_position(self, ticker: str) -> dict:
        try:
            self._delete(f"/v2/positions/{ticker.upper()}")
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}

    def held_tickers(self) -> set:
        return {p["ticker"] for p in self.get_positions()}

    # ══════════════════════════════════════════════════════════════════════════
    # OPTIONS (paper) — requires options enabled on the account (Level 2 to buy)
    # ══════════════════════════════════════════════════════════════════════════
    def options_status(self) -> dict:
        """
        Detect whether options trading is enabled + the approved level.
        Returns {enabled, level, buying_power, msg}.
        """
        try:
            a = self._get("/v2/account")
            lvl = a.get("options_trading_level")
            lvl = int(lvl) if lvl is not None else 0
            obp = float(a.get("options_buying_power", 0) or 0)
            enabled = lvl >= 1
            can_buy = lvl >= 2          # long calls/puts need Level 2
            if not enabled:
                msg = ("Options NOT enabled on this paper account. In the Alpaca "
                       "dashboard → account config → enable Options (Level 2) to "
                       "buy calls/puts. Instant on paper.")
            elif not can_buy:
                msg = (f"Options enabled at Level {lvl}, but buying calls/puts needs "
                       f"Level 2. Raise the options level in the Alpaca dashboard.")
            else:
                msg = f"Options enabled — Level {lvl}, buying power ${obp:,.0f}."
            return {"enabled": enabled, "can_buy_longs": can_buy,
                    "level": lvl, "buying_power": obp, "msg": msg}
        except Exception as e:
            return {"enabled": False, "can_buy_longs": False, "level": 0,
                    "buying_power": 0, "msg": f"error: {str(e)[:160]}"}

    def find_option_contract(self, underlying: str, expiry: str,
                             opt_type: str, target_strike: float) -> dict | None:
        """
        Look up the canonical Alpaca OCC symbol for the contract closest to
        target_strike on the given expiry. Returns {symbol, strike, tradable,
        close_price} or None. Uses Alpaca's contracts endpoint so symbols always
        match what the broker will accept.
        """
        try:
            params = {
                "underlying_symbols": underlying.upper(),
                "expiration_date": expiry,
                "type": "call" if opt_type.lower().startswith("c") else "put",
                "status": "active", "limit": 200,
            }
            data = self._get("/v2/options/contracts", params)
            contracts = data.get("option_contracts", []) or []
            if not contracts:
                return None
            best, bd = None, 1e18
            for c in contracts:
                try:
                    sk = float(c.get("strike_price", 0))
                except Exception:
                    continue
                d = abs(sk - target_strike)
                if d < bd and c.get("tradable", True):
                    bd, best = d, c
            if not best:
                return None
            return {
                "symbol": best.get("symbol"),
                "strike": float(best.get("strike_price", 0)),
                "tradable": bool(best.get("tradable", True)),
                "expiry": best.get("expiration_date"),
                "close_price": float(best.get("close_price") or 0) or None,
            }
        except Exception:
            return None

    def submit_option_buy(self, occ_symbol: str, qty: int = 1) -> dict:
        """Market buy-to-open `qty` contracts of an option (paper)."""
        if not self.available():
            return {"ok": False, "error": "broker not configured"}
        st = self.options_status()
        if not st["can_buy_longs"]:
            return {"ok": False, "error": st["msg"]}
        body = {"symbol": occ_symbol, "qty": str(int(qty)), "side": "buy",
                "type": "market", "time_in_force": "day"}
        try:
            o = self._post("/v2/orders", body)
            return {"ok": True, "order_id": o.get("id"), "qty": int(qty),
                    "symbol": occ_symbol, "status": o.get("status")}
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}

    def close_option(self, occ_symbol: str) -> dict:
        try:
            self._delete(f"/v2/positions/{occ_symbol}")
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}

    def get_option_positions(self) -> list:
        """Open OPTION positions (asset_class == us_option), with live P&L."""
        try:
            rows = self._get("/v2/positions")
        except Exception:
            return []
        out = []
        for p in rows:
            if p.get("asset_class") != "us_option":
                continue
            try:
                out.append({
                    "symbol": p.get("symbol"),
                    "underlying": p.get("symbol", "")[:6].rstrip("0123456789"),
                    "qty": float(p.get("qty", 0)),
                    "avg_entry": float(p.get("avg_entry_price", 0)),
                    "current": float(p.get("current_price", 0) or 0),
                    "market_value": float(p.get("market_value", 0) or 0),
                    "cost_basis": float(p.get("cost_basis", 0) or 0),
                    "unrealized_pnl": float(p.get("unrealized_pl", 0) or 0),
                    "unrealized_pct": round(float(p.get("unrealized_plpc", 0) or 0) * 100, 2),
                })
            except Exception:
                continue
        return out

    def held_option_underlyings(self) -> set:
        return {p["underlying"] for p in self.get_option_positions()}

    # ── OPTIONS EXIT MANAGER — autonomous position management ──────────────────
    @staticmethod
    def parse_occ_symbol(symbol: str) -> dict | None:
        """
        Parse an OCC option symbol like 'NVDA260612C00212000' into
        {underlying, expiry (date), type, strike}. Format from the end:
        [8-digit strike (price*1000)][1 char C/P][6-digit YYMMDD][underlying].
        """
        from datetime import datetime as _dt
        try:
            s = symbol.strip().upper()
            strike = int(s[-8:]) / 1000.0
            opt_type = "call" if s[-9] == "C" else "put"
            ymd = s[-15:-9]
            expiry = _dt.strptime(ymd, "%y%m%d").date()
            underlying = s[:-15]
            return {"underlying": underlying, "expiry": expiry,
                    "type": opt_type, "strike": strike}
        except Exception:
            return None

    def manage_option_exits(self, tp_pct: float = 100.0, sl_pct: float = -50.0,
                            dte_floor: int = 1) -> list:
        """
        Autonomous exit rules for every open option position:
          • close at >= tp_pct unrealized   (take profit, default +100%)
          • close at <= sl_pct unrealized   (stop loss, default -50%)
          • close at <= dte_floor days left  (theta-cliff time stop)
        Returns a list of {symbol, underlying, reason, pct, pnl} for each close.
        """
        from datetime import date as _date
        closed = []
        for p in self.get_option_positions():
            sym = p["symbol"]
            pct = p["unrealized_pct"]
            pnl = p["unrealized_pnl"]
            parsed = self.parse_occ_symbol(sym)
            dte = None
            if parsed:
                try:
                    dte = (parsed["expiry"] - _date.today()).days
                except Exception:
                    dte = None

            reason = None
            if pct >= tp_pct:
                reason = "TAKE_PROFIT"
            elif pct <= sl_pct:
                reason = "STOP_LOSS"
            elif dte is not None and dte <= dte_floor:
                reason = "TIME_STOP"

            if reason:
                res = self.close_option(sym)
                if res.get("ok"):
                    closed.append({"symbol": sym,
                                   "underlying": p.get("underlying", ""),
                                   "reason": reason, "pct": pct, "pnl": pnl,
                                   "dte": dte})
        return closed


if __name__ == "__main__":
    b = AlpacaPaperBroker()
    t = b.test_connection()
    print("Connection:", t)
    if t.get("ok"):
        print("Account:", b.get_account())
        print("Positions:", len(b.get_positions()))
        print("Recent orders:", len(b.get_orders(limit=10)))
    else:
        print("\nTo enable: create a free Alpaca paper account, then set "
              "ALPACA_PAPER_KEY and ALPACA_PAPER_SECRET (env / Streamlit "
              "secrets / GitHub secrets).")
