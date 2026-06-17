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
DATA_BASE  = "https://data.alpaca.markets"   # market data (option quotes)


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

    def realized_pnl_from_fills(self, limit: int = 500) -> dict:
        """
        GROUND TRUTH realized P&L computed from actual filled orders (not the
        journal's scan-estimated premiums, which under-counted churn ~8x on
        2026-06-12). Pairs buy/sell fills per symbol: realized = sell proceeds -
        buy cost for fully round-tripped quantity. Options use a 100x multiplier.

        Returns {realized_pnl, round_trips, by_symbol:{sym:{buys,sells,pnl,trips}},
                 churn:[(sym, trips)]}.
        """
        try:
            orders = self._get("/v2/orders", {"status": "closed", "limit": limit,
                                              "direction": "desc"})
        except Exception:
            return {"realized_pnl": 0.0, "round_trips": 0, "by_symbol": {}, "churn": []}
        agg: dict = {}
        for o in orders:
            if str(o.get("status")) != "filled":
                continue
            sym = o.get("symbol", "")
            fp = float(o.get("filled_avg_price", 0) or 0)
            fq = float(o.get("filled_qty", 0) or 0)
            if fp <= 0 or fq <= 0:
                continue
            mult = 100.0 if len(sym) > 8 else 1.0      # OCC option symbols are long
            notional = fp * fq * mult
            a = agg.setdefault(sym, {"buys": 0.0, "sells": 0.0,
                                     "buy_ct": 0, "sell_ct": 0})
            if o.get("side") == "buy":
                a["buys"] += notional; a["buy_ct"] += 1
            else:
                a["sells"] += notional; a["sell_ct"] += 1
        by_symbol, total, churn = {}, 0.0, []
        for sym, a in agg.items():
            trips = min(a["buy_ct"], a["sell_ct"])
            pnl = a["sells"] - a["buys"]               # only meaningful if flat
            by_symbol[sym] = {"buys": round(a["buys"], 2), "sells": round(a["sells"], 2),
                              "pnl": round(pnl, 2), "trips": trips}
            total += pnl
            if trips >= 3:
                churn.append((sym, trips))
        churn.sort(key=lambda x: -x[1])
        return {"realized_pnl": round(total, 2),
                "round_trips": sum(min(a["buy_ct"], a["sell_ct"]) for a in agg.values()),
                "by_symbol": by_symbol, "churn": churn}

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

    def submit_notional_buy(self, ticker: str, notional: float) -> dict:
        """
        Plain market buy by DOLLAR amount (fractional shares, no bracket).
        Used by the Overnight Edge: buy the close, sell at the open — exits are
        time-based, not price-based, so no stop/target is attached.
        Notional orders must be time_in_force='day' on Alpaca.
        """
        if not self.available():
            return {"ok": False, "error": "broker not configured"}
        body = {
            "symbol": ticker.upper(), "notional": str(round(notional, 2)),
            "side": "buy", "type": "market", "time_in_force": "day",
        }
        try:
            o = self._post("/v2/orders", body)
            return {"ok": True, "order_id": o.get("id"),
                    "notional": round(notional, 2)}
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

    def get_option_quote(self, occ_symbol: str) -> dict | None:
        """
        Live bid/ask for one option contract from Alpaca market data.
        Returns {bid, ask, mid, spread_pct} or None if no quote.
        """
        if not self.available():
            return None
        try:
            r = requests.get(f"{DATA_BASE}/v1beta1/options/quotes/latest",
                             headers=self._headers(),
                             params={"symbols": occ_symbol}, timeout=10)
            r.raise_for_status()
            q = (r.json().get("quotes", {}) or {}).get(occ_symbol, {})
            bid = float(q.get("bp", 0) or 0)
            ask = float(q.get("ap", 0) or 0)
            if bid <= 0 or ask <= 0 or ask < bid:
                return None
            mid = (bid + ask) / 2.0
            return {"bid": bid, "ask": ask, "mid": round(mid, 4),
                    "spread_pct": round((ask - bid) / mid * 100, 1) if mid > 0 else 999}
        except Exception:
            return None

    def submit_option_buy(self, occ_symbol: str, qty: int = 1,
                          limit_price: float | None = None) -> dict:
        """
        Buy-to-open `qty` contracts. If `limit_price` is given, submit a LIMIT
        order (caps slippage — the 2026-06-12 lesson: a market order on a wide
        spread filled a $1.30-estimated contract at $4.80). Falls back to market
        only when no limit is provided.
        """
        if not self.available():
            return {"ok": False, "error": "broker not configured"}
        st = self.options_status()
        if not st["can_buy_longs"]:
            return {"ok": False, "error": st["msg"]}
        body = {"symbol": occ_symbol, "qty": str(int(qty)), "side": "buy",
                "time_in_force": "day"}
        if limit_price and limit_price > 0:
            body["type"] = "limit"
            body["limit_price"] = str(round(float(limit_price), 2))
        else:
            body["type"] = "market"
        try:
            o = self._post("/v2/orders", body)
            return {"ok": True, "order_id": o.get("id"), "qty": int(qty),
                    "symbol": occ_symbol, "status": o.get("status"),
                    "order_type": body["type"],
                    "limit_price": body.get("limit_price")}
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}

    def submit_option_spread(self, long_symbol: str, short_symbol: str,
                             qty: int = 1, net_debit_limit: float | None = None) -> dict:
        """Buy-to-open a DEBIT (bull-call) spread as ONE multi-leg (mleg) order.

        long_symbol  — the near-money call we BUY  (buy_to_open)
        short_symbol — the further-OTM call we SELL (sell_to_open), SAME expiry
        net_debit_limit — max NET debit per spread ($/share). Submitted as an mleg
        LIMIT so we never pay more than the modeled debit + buffer; market only if
        no limit is given. Alpaca fills/cancels both legs atomically, so there is no
        legging risk. Requires options level 3 (spreads) — guarded by options_status.

        Returns {ok, order_id, ...} shaped like submit_option_buy().
        """
        if not self.available():
            return {"ok": False, "error": "broker not configured"}
        st = self.options_status()
        if not st["can_buy_longs"]:
            return {"ok": False, "error": st["msg"]}
        legs = [
            {"symbol": long_symbol,  "ratio_qty": "1", "side": "buy",
             "position_intent": "buy_to_open"},
            {"symbol": short_symbol, "ratio_qty": "1", "side": "sell",
             "position_intent": "sell_to_open"},
        ]
        body = {"order_class": "mleg", "qty": str(int(qty)),
                "time_in_force": "day", "legs": legs}
        if net_debit_limit and net_debit_limit > 0:
            body["type"] = "limit"
            body["limit_price"] = str(round(float(net_debit_limit), 2))
        else:
            body["type"] = "market"
        try:
            o = self._post("/v2/orders", body)
            return {"ok": True, "order_id": o.get("id"), "qty": int(qty),
                    "symbol": f"{long_symbol}/{short_symbol}",
                    "status": o.get("status"), "order_type": body["type"],
                    "limit_price": body.get("limit_price"), "order_class": "mleg"}
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}

    def close_option(self, occ_symbol: str) -> dict:
        try:
            self._delete(f"/v2/positions/{occ_symbol}")
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}

    def cancel_stale_orders(self, older_than_min: float = 5.0) -> int:
        """
        Cancel still-open orders older than `older_than_min` minutes. A marketable
        limit should fill in seconds, so anything still open at the next 10-min
        cycle didn't fill — leaving it risks a surprise late fill and ties up
        buying power. Returns the count canceled. (Added with the limit-order
        switch so unfilled limits can't linger over an unattended week.)
        """
        if not self.available():
            return 0
        from datetime import datetime as _dt, timezone as _tz
        now = _dt.now(_tz.utc)
        n = 0
        try:
            orders = self._get("/v2/orders", {"status": "open", "limit": 100})
        except Exception:
            return 0
        for o in orders or []:
            try:
                sub = str(o.get("submitted_at") or o.get("created_at") or "")
                if sub:
                    ts = _dt.fromisoformat(sub.replace("Z", "+00:00"))
                    if (now - ts).total_seconds() < older_than_min * 60:
                        continue            # too fresh — give it a chance to fill
                self._delete(f"/v2/orders/{o.get('id')}")
                n += 1
            except Exception:
                continue
        return n

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

    def manage_option_exits(self, conn=None,
                            activation_pct: float = 50.0, trail_frac: float = 0.30,
                            hard_stop_pct: float = -50.0, dte_floor: int = 2,
                            put_activation: float = 40.0, put_trail_frac: float = 0.25,
                            put_hard_stop: float = -45.0,
                            grace_minutes: float = 60.0,
                            grace_hard_stop: float = -80.0) -> list:
        """
        TRAILING-STOP exit manager — lets winners RUN (no take-profit cap) while
        protecting gains. This is the answer to 'don't sell a +300% runner at +70%.'

        Logic per open option (direction parsed from the OCC symbol):
          • Track the PEAK premium reached (persisted in broker_option_peaks).
          • Before the trade reaches `activation_pct` (+50% calls / +40% puts):
            hard stop only (-50% / -45%). Losers cut fast, no early profit-taking.
          • Once activated: switch to a TRAILING stop at `trail_frac` below the
            peak premium (30% calls / 25% puts). The stop RISES as the option
            makes new highs, so it stays in the whole run and only exits on a
            real pullback from the top — capturing the full move, capped by
            nothing on the upside.
          • Always: DTE<=floor time-stop (theta cliff).

        GRACE PERIOD (added after the 2026-06-12 CPI whipsaw, where fresh
        positions were guillotined at the opening low by the -50% hard stop, then
        the exact contracts ran +58% to +177%): for the first `grace_minutes` of a
        position's life the hard stop is RELAXED to `grace_hard_stop` (-80%), so
        normal opening option-leverage noise can't shake us out before the thesis
        has a chance to work. Trailing stop and time stop still apply.

        conn (optional) persists the peak per contract across the 5-min cycles.
        Returns a list of {symbol, underlying, reason, pct, pnl, peak_pct}.
        """
        from datetime import date as _date, datetime as _dt, timezone as _tz
        _now_utc = _dt.now(_tz.utc)
        if conn is not None:
            try:
                conn.execute("CREATE TABLE IF NOT EXISTS broker_option_peaks "
                             "(contract_symbol TEXT PRIMARY KEY, peak_premium REAL, "
                             "peak_pct REAL, activated INTEGER DEFAULT 0)")
                # add first_seen column if missing (proxy for position age)
                try:
                    conn.execute("ALTER TABLE broker_option_peaks "
                                 "ADD COLUMN first_seen TEXT")
                except Exception:
                    pass
            except Exception:
                conn = None

        def _peak_read(sym):
            if conn is None:
                return None
            try:
                r = conn.execute("SELECT peak_premium, peak_pct, activated, first_seen "
                                 "FROM broker_option_peaks WHERE contract_symbol=?",
                                 (sym,)).fetchone()
                if not r:
                    return None
                g = (lambda k, i: r.get(k) if hasattr(r, "get") else r[i])
                return (float(g("peak_premium", 0) or 0), float(g("peak_pct", 0) or 0),
                        int(g("activated", 0) or 0), g("first_seen", 3))
            except Exception:
                return None

        def _peak_write(sym, prem, pct, activated, first_seen):
            if conn is None:
                return
            try:
                conn.execute("INSERT INTO broker_option_peaks "
                             "(contract_symbol, peak_premium, peak_pct, activated, first_seen) "
                             "VALUES (?,?,?,?,?) ON CONFLICT(contract_symbol) DO UPDATE SET "
                             "peak_premium=excluded.peak_premium, peak_pct=excluded.peak_pct, "
                             "activated=excluded.activated",
                             (sym, prem, pct, int(activated), first_seen))
            except Exception:
                pass

        def _peak_clear(sym):
            if conn is None:
                return
            try:
                conn.execute("DELETE FROM broker_option_peaks WHERE contract_symbol=?", (sym,))
            except Exception:
                pass

        # ── SPREAD AWARENESS — manage each debit spread as ONE unit ─────────────
        # broker_spread_legs (written by monitor.py when OPTION_STRUCTURE=spread)
        # maps a long call -> its short hedge. When the table is empty (the
        # default 'naked' state) every spread branch below is skipped and the
        # naked-call exit path is byte-for-byte unchanged.
        _long_pair = {}    # long_symbol -> (short_symbol, entry_debit_per_share)
        _short_set = set()  # short legs: never stopped on their own
        if conn is not None:
            try:
                conn.execute("CREATE TABLE IF NOT EXISTS broker_spread_legs "
                             "(long_symbol TEXT PRIMARY KEY, short_symbol TEXT, "
                             "underlying TEXT, entry_debit REAL, opened_at TEXT)")
                for r in conn.execute("SELECT long_symbol, short_symbol, "
                                      "entry_debit FROM broker_spread_legs").fetchall():
                    try:
                        _ls, _ss, _ed = (r["long_symbol"], r["short_symbol"],
                                         r["entry_debit"])
                    except Exception:
                        _ls, _ss, _ed = r[0], r[1], r[2]
                    if _ls and _ss:
                        _long_pair[_ls] = (_ss, float(_ed or 0) or 0.0)
                        _short_set.add(_ss)
            except Exception:
                _long_pair, _short_set = {}, set()

        positions = self.get_option_positions()
        _by_sym = {p["symbol"]: p for p in positions}

        # ORPHAN-SHORT CLEANUP: if a spread's long leg is already gone but the
        # short hedge is still open, buy it back NOW so we are never left holding
        # a naked SHORT call (unbounded risk).
        for _ls, (_ss, _ed) in list(_long_pair.items()):
            if _ls not in _by_sym and _ss in _by_sym:
                try:
                    self.close_option(_ss)
                except Exception:
                    pass
                _peak_clear(_ss)
                if conn is not None:
                    try:
                        conn.execute("DELETE FROM broker_spread_legs "
                                     "WHERE long_symbol=?", (_ls,))
                    except Exception:
                        pass

        closed = []
        for p in positions:
            sym = p["symbol"]
            if sym in _short_set:
                continue  # short leg is closed together with its long leg
            _spread_short = None
            if sym in _long_pair:
                # SPREAD: value, %, and P&L are computed on the NET (long premium
                # minus short premium); the pair is stopped and closed as one unit.
                _ss, _ed = _long_pair[sym]
                _sp = _by_sym.get(_ss)
                _long_cur = p.get("current", 0) or 0
                _short_cur = (_sp.get("current", 0) or 0) if _sp else 0.0
                cur_prem = _long_cur - _short_cur
                _qty = abs(p.get("qty", 0) or 0) or 1
                pct = ((cur_prem / _ed - 1) * 100) if _ed > 0 else 0.0
                pnl = (cur_prem - _ed) * 100 * _qty
                _spread_short = _ss
            else:
                pct = p["unrealized_pct"]
                pnl = p["unrealized_pnl"]
                cur_prem = p.get("current", 0) or 0
            # GUARD: if Alpaca reports no/zero price (illiquid contract, thin
            # quotes, before-open), do NOT make an exit decision — a 0 price would
            # look like -100% and falsely trigger a stop. Skip until a real quote.
            if cur_prem <= 0:
                continue
            parsed = self.parse_occ_symbol(sym)
            is_put = bool(parsed and parsed.get("type") == "put")
            dte = None
            if parsed:
                try:
                    dte = (parsed["expiry"] - _date.today()).days
                except Exception:
                    dte = None

            _act = put_activation if is_put else activation_pct
            _trail = put_trail_frac if is_put else trail_frac
            _hard = put_hard_stop if is_put else hard_stop_pct
            _dte_floor = 2 if is_put else dte_floor

            # update peak (+ record first_seen on first sighting)
            prev = _peak_read(sym)
            peak_prem = max(prev[0] if prev else 0, cur_prem)
            peak_pct = max(prev[1] if prev else -999, pct)
            activated = bool((prev[2] if prev else 0)) or (peak_pct >= _act)
            _first_seen = (prev[3] if (prev and prev[3]) else _now_utc.isoformat())
            _peak_write(sym, peak_prem, peak_pct, activated, _first_seen)

            # position age (minutes) — within grace, relax the hard stop so opening
            # volatility on a fresh cheap option can't guillotine a good thesis.
            _age_min = 10**9
            try:
                _fs = _dt.fromisoformat(str(_first_seen).replace("Z", "+00:00"))
                if _fs.tzinfo is None:
                    _fs = _fs.replace(tzinfo=_tz.utc)
                _age_min = (_now_utc - _fs).total_seconds() / 60.0
            except Exception:
                pass
            _in_grace = _age_min < grace_minutes
            _eff_hard = grace_hard_stop if _in_grace else _hard

            reason = None
            if dte is not None and dte <= _dte_floor:
                reason = "TIME_STOP"
            elif not activated:
                if pct <= _eff_hard:
                    reason = "STOP_LOSS"
            else:
                # trailing: stop = peak premium minus trail fraction
                trail_stop = peak_prem * (1 - _trail)
                if cur_prem <= trail_stop and peak_prem > 0:
                    reason = "TRAILING_STOP"

            if reason:
                res = self.close_option(sym)
                if res.get("ok"):
                    _peak_clear(sym)
                    if _spread_short is not None:
                        # close the short hedge in the same cycle and clear the
                        # pairing so the spread leaves the book as one unit.
                        try:
                            self.close_option(_spread_short)
                        except Exception:
                            pass
                        _peak_clear(_spread_short)
                        if conn is not None:
                            try:
                                conn.execute("DELETE FROM broker_spread_legs "
                                             "WHERE long_symbol=?", (sym,))
                            except Exception:
                                pass
                    closed.append({"symbol": (f"{sym}/{_spread_short}"
                                              if _spread_short else sym),
                                   "underlying": p.get("underlying", ""),
                                   "reason": reason, "pct": pct, "pnl": pnl,
                                   "peak_pct": round(peak_pct, 0), "dte": dte})
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
