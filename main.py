
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — Crypto SPOT trend screener (OKX) + Telegram alert

Logic:
1) Trend Template (crypto-adjusted Minervini 8 rules, EMA + 180D high/low)
2) Accumulation filter (VCP-lite via ATR contraction and/or Darvas Box range)
3) Momentum confirmation (RSI, ADX, Volume, BB width)
4) Send qualifying coins to Telegram

Environment variables:
- TELEGRAM_BOT_TOKEN: Telegram bot token
- TELEGRAM_CHAT_ID: Telegram chat id (group/user)
- OKX_BASE_URL (optional): default https://www.okx.com
- MAX_COINS (optional): limit number of spot USDT pairs to scan (default None = all live pairs)
- TIMEFRAME (optional): kline timeframe, default 1D
- MIN_USDT_PRICE (optional): filter only coins with last price < this value (e.g., 1.0). Default: no cap
- MIN_24H_USDT_VOL (optional): minimum 24h quote volume in USDT (default 1_000_000); requires market/tickers call
- TOP_N (optional): top N to send (by momentum score). Default 30

Run:
$ python3 main.py

Author: EROS/ChatGPT
"""

import os
import sys
import time
import math
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import requests

# ------------------------------ Config ------------------------------

OKX_BASE_URL = os.getenv("OKX_BASE_URL", "https://www.okx.com")
TIMEFRAME = os.getenv("TIMEFRAME", "1D")  # OKX bars (e.g., 1D, 4H, 1H)
MAX_COINS = int(os.getenv("MAX_COINS")) if os.getenv("MAX_COINS") else None
MIN_USDT_PRICE = float(os.getenv("MIN_USDT_PRICE")) if os.getenv("MIN_USDT_PRICE") else None
MIN_24H_USDT_VOL = float(os.getenv("MIN_24H_USDT_VOL", "1000000"))
TOP_N = int(os.getenv("TOP_N", "30"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Trend Template thresholds (crypto adjusted)
PCT_ABOVE_LOW_WINDOW_DAYS = 180   # replace "52 weeks"
PCT_ABOVE_LOW_MIN = 0.30          # >= 30% above 180D low
PCT_BELOW_HIGH_MAX = 0.25         # <= 25% below 180D high

# VCP-lite & Darvas thresholds
ATR_WINDOW = 20
ATR_CONTRACTION_LOOKBACK = 30           # lookback to confirm contraction
ATR_CONTRACTION_RATIO_MAX = 0.08        # ATR20 / close <= 4%
DARVAS_WINDOW = 15                       # box length
DARVAS_MAX_WIDTH = 0.10                  # box height relative to price <= 10%

# Momentum thresholds
RSI_MIN = 50                            # RSI >= 55 
ADX_MIN = 20
VOL_SPIKE_MULT = 1.2
BB_WIDTH_MAX = 0.1                      # BB width <= 6% of price

# Candles
CANDLE_LIMIT = 300  # enough for EMA200 on daily


# ------------------------------ Utilities ------------------------------

def http_get(url: str, params: Dict[str, Any], retries: int = 3, timeout: int = 15) -> Optional[Dict[str, Any]]:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            else:
                logging.warning(f"GET {url} status={r.status_code} attempt={attempt} params={params}")
        except Exception as e:
            logging.warning(f"GET {url} exception={e} attempt={attempt} params={params}")
        time.sleep(0.8 * attempt)
    return None


def ema(values: List[float], period: int) -> List[float]:
    if not values or period <= 0:
        return []
    k = 2 / (period + 1)
    res = []
    ema_prev = None
    for v in values:
        if ema_prev is None:
            ema_prev = v
        else:
            ema_prev = v * k + ema_prev * (1 - k)
        res.append(ema_prev)
    return res


def sma(values: List[float], period: int) -> List[float]:
    out = []
    s = 0.0
    q = []
    for v in values:
        q.append(v)
        s += v
        if len(q) > period:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def rsi(values: List[float], period: int = 14) -> List[float]:
    if len(values) < period + 1:
        return [math.nan] * len(values)
    gains = []
    losses = []
    for i in range(1, period + 1):
        ch = values[i] - values[i - 1]
        gains.append(max(ch, 0))
        losses.append(max(-ch, 0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    out = [math.nan] * period
    if avg_loss == 0:
        out.append(100.0)
    else:
        rs = avg_gain / avg_loss
        out.append(100 - (100 / (1 + rs)))
    for i in range(period + 1, len(values)):
        ch = values[i] - values[i - 1]
        gain = max(ch, 0)
        loss = max(-ch, 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            out.append(100.0)
        else:
            rs = avg_gain / avg_loss
            out.append(100 - (100 / (1 + rs)))
    return out


def true_range(h: List[float], l: List[float], c: List[float]) -> List[float]:
    trs = []
    prev_close = None
    for i in range(len(c)):
        if prev_close is None:
            tr = h[i] - l[i]
        else:
            tr = max(h[i] - l[i], abs(h[i] - prev_close), abs(l[i] - prev_close))
        trs.append(tr)
        prev_close = c[i]
    return trs


def atr(high: List[float], low: List[float], close: List[float], period: int = 14) -> List[float]:
    trs = true_range(high, low, close)
    return sma(trs, period)


def adx(high: List[float], low: List[float], close: List[float], period: int = 14) -> Tuple[List[float], List[float], List[float]]:
    # Returns: ADX, +DI, -DI
    if len(close) < period + 2:
        n = len(close)
        return [math.nan]*n, [math.nan]*n, [math.nan]*n
    plus_dm = [0.0]
    minus_dm = [0.0]
    for i in range(1, len(high)):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        plus_dm.append(max(up, 0.0) if up > down and up > 0 else 0.0)
        minus_dm.append(max(down, 0.0) if down > up and down > 0 else 0.0)

    atr_vals = atr(high, low, close, period)
    plus_di = []
    minus_di = []
    for i in range(len(close)):
        if atr_vals[i] and atr_vals[i] != 0:
            plus_di.append(100 * (sma(plus_dm, period)[i] / atr_vals[i]))
            minus_di.append(100 * (sma(minus_dm, period)[i] / atr_vals[i]))
        else:
            plus_di.append(math.nan)
            minus_di.append(math.nan)

    dx = []
    for i in range(len(close)):
        if math.isnan(plus_di[i]) or math.isnan(minus_di[i]) or (plus_di[i] + minus_di[i]) == 0:
            dx.append(math.nan)
        else:
            dx.append(100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]))

    adx_vals = sma(dx, period)
    return adx_vals, plus_di, minus_di


def bbands(values: List[float], period: int = 20, std_mult: float = 2.0) -> Tuple[List[float], List[float], List[float], List[float]]:
    ma = sma(values, period)
    stds = []
    from collections import deque
    dq = deque()
    s, s2 = 0.0, 0.0
    for v in values:
        dq.append(v)
        s += v
        s2 += v*v
        if len(dq) > period:
            old = dq.popleft()
            s -= old
            s2 -= old*old
        n = len(dq)
        mean = s / n
        var = max(s2 / n - mean*mean, 0.0)
        stds.append(math.sqrt(var))
    upper = [ma[i] + std_mult * stds[i] for i in range(len(values))]
    lower = [ma[i] - std_mult * stds[i] for i in range(len(values))]
    width = [ (upper[i] - lower[i]) / values[i] if values[i] != 0 else math.nan for i in range(len(values)) ]
    return ma, upper, lower, width


# ------------------------------ OKX API ------------------------------

def okx_get_instruments_spot_usdt() -> List[Dict[str, Any]]:
    url = f"{OKX_BASE_URL}/api/v5/public/instruments"
    data = http_get(url, {"instType": "SPOT"})
    if not data or data.get("code") != "0":
        logging.error("Failed to fetch instruments")
        return []
    items = data.get("data", [])
    out = []
    for it in items:
        if it.get("quoteCcy") == "USDT" and it.get("state") == "live":
            out.append(it)
    return out


def okx_get_tickers_spot() -> Dict[str, Dict[str, Any]]:
    """Return dict instId -> ticker for 24h volume filter and last price."""
    url = f"{OKX_BASE_URL}/api/v5/market/tickers"
    data = http_get(url, {"instType": "SPOT"})
    res = {}
    if data and data.get("code") == "0":
        for it in data.get("data", []):
            res[it["instId"]] = it
    return res


def okx_get_candles(instId: str, bar: str = TIMEFRAME, limit: int = CANDLE_LIMIT) -> Optional[List[List[str]]]:
    url = f"{OKX_BASE_URL}/api/v5/market/candles"
    data = http_get(url, {"instId": instId, "bar": bar, "limit": limit})
    if not data or data.get("code") != "0":
        return None
    # OKX returns newest-first, we want oldest-first
    arr = data.get("data", [])
    arr = list(reversed(arr))
    return arr


# ------------------------------ Screening Logic ------------------------------

@dataclass
class CandleData:
    ts: List[int]
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    vol: List[float]
    vol_quote: List[float]


def parse_okx_candles(raw: List[List[str]]) -> CandleData:
    # Each item: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    ts, o, h, l, c, vol, vol_quote = [], [], [], [], [], [], []
    for row in raw:
        ts.append(int(row[0]))
        o.append(float(row[1]))
        h.append(float(row[2]))
        l.append(float(row[3]))
        c.append(float(row[4]))
        vol.append(float(row[5]))
        # OKX sometimes puts quote vol in row[7] (volCcyQuote) or row[6] (volCcy)
        # We'll try both safely.
        qv = None
        try:
            qv = float(row[7])
        except Exception:
            try:
                qv = float(row[6])
            except Exception:
                qv = 0.0
        vol_quote.append(qv if qv is not None else 0.0)
    return CandleData(ts, o, h, l, c, vol, vol_quote)


def pct(x: float) -> str:
    return f"{x*100:.1f}%"


def trend_template_crypto(c: List[float]) -> Dict[str, bool]:
    # Requires EMA50/150/200 + 180D high/low proximity
    ema50 = ema(c, 50)
    ema150 = ema(c, 150)
    ema200 = ema(c, 200)
    price = c[-1]

    conds = {}
    conds["close_gt_ema150"] = price > ema150[-1]
    conds["close_gt_ema200"] = price > ema200[-1]
    conds["ema150_gt_ema200"] = ema150[-1] > ema200[-1]

    # slope of EMA200 positive (approx)
    ema200_slope = ema200[-1] - ema200[-5] if len(ema200) > 5 else 0.0
    conds["ema200_trending_up"] = ema200_slope > 0

    conds["ema50_stack"] = ema(ema50, 1)[-1] > ema150[-1] and ema50[-1] > ema200[-1]
    conds["close_gt_ema50"] = price > ema50[-1]

    # 180D high/low
    w = min(PCT_ABOVE_LOW_WINDOW_DAYS, len(c))
    win_slice = c[-w:]
    low_w = min(win_slice)
    high_w = max(win_slice)
    conds["pct_above_low"] = (price - low_w) / low_w >= PCT_ABOVE_LOW_MIN if low_w > 0 else False
    conds["pct_below_high"] = (high_w - price) / high_w <= PCT_BELOW_HIGH_MAX if high_w > 0 else False

    conds["all_ok"] = all(conds.values())
    return conds


def vcp_lite_filter(high: List[float], low: List[float], close: List[float]) -> bool:
    # ATR contraction and normalized ATR small
    atr20 = atr(high, low, close, ATR_WINDOW)
    if len(atr20) < ATR_CONTRACTION_LOOKBACK + 5:
        return False
    recent = atr20[-ATR_CONTRACTION_LOOKBACK:]
    if any(math.isnan(x) for x in recent):
        return False
    # slope negative across the window (approx via start-end)
    slope = recent[-1] - recent[0]
    norm = recent[-1] / close[-1] if close[-1] > 0 else 1.0
    return (slope < 0) and (norm <= ATR_CONTRACTION_RATIO_MAX)


def darvas_box_filter(high: List[float], low: List[float], close: List[float]) -> Tuple[bool, Optional[float]]:
    # Use last DARVAS_WINDOW bars as a "box"
    if len(close) < DARVAS_WINDOW + 2:
        return False, None
    box_high = max(high[-DARVAS_WINDOW:])
    box_low = min(low[-DARVAS_WINDOW:])
    box_width = (box_high - box_low) / close[-1] if close[-1] > 0 else 1.0
    # Breakout condition: last close > box_high by small margin
    broke = close[-1] > box_high * 1.005  # 0.5% buffer
    tight = box_width <= DARVAS_MAX_WIDTH
    return (tight and broke), box_high


def momentum_confirm(close: List[float], high: List[float], low: List[float], vol: List[float]) -> Dict[str, bool]:
    rsi_vals = rsi(close, 14)
    adx_vals, plus_di, minus_di = adx(high, low, close, 14)
    ma20, bb_u, bb_l, bb_w = bbands(close, 20, 2.0)
    vol_sma20 = sma(vol, 20)
    conds = {
        "rsi": rsi_vals[-1] >= RSI_MIN if not math.isnan(rsi_vals[-1]) else False,
        "adx": adx_vals[-1] >= ADX_MIN if not math.isnan(adx_vals[-1]) else False,
        "di": (plus_di[-1] > minus_di[-1]) if not (math.isnan(plus_di[-1]) or math.isnan(minus_di[-1])) else False,
        "vol_spike": (vol[-1] >= vol_sma20[-1] * VOL_SPIKE_MULT) if vol_sma20[-1] > 0 else False,
        "bb_width": (bb_w[-1] <= BB_WIDTH_MAX) if not math.isnan(bb_w[-1]) else False
    }
    conds["all_ok"] = all(conds.values())
    return conds


def momentum_score(close: List[float], high: List[float], low: List[float], vol: List[float]) -> float:
    # Simple composite score to rank results
    rsi_vals = rsi(close, 14)
    adx_vals, plus_di, minus_di = adx(high, low, close, 14)
    ma20, bb_u, bb_l, bb_w = bbands(close, 20, 2.0)
    vol_sma20 = sma(vol, 20)
    score = 0.0
    last = -1
    if not math.isnan(rsi_vals[last]):
        score += (rsi_vals[last] - 50) / 25  # 0..2 range approx
    if not math.isnan(adx_vals[last]):
        score += (adx_vals[last] - 20) / 20  # 0..2
    if not math.isnan(bb_w[last]) and bb_w[last] > 0:
        score += max(0.0, (BB_WIDTH_MAX / bb_w[last])) * 0.5  # tighter bands better
    if vol_sma20[last] > 0:
        score += min(2.0, vol[last] / vol_sma20[last]) * 0.5
    return round(score, 3)


# ------------------------------ Telegram ------------------------------

def tg_send_message(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.error("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=20)
        if r.status_code == 200:
            return True
        logging.error(f"Telegram send failed: {r.status_code} {r.text}")
    except Exception as e:
        logging.error(f"Telegram send exception: {e}")
    return False


# ------------------------------ Main Flow ------------------------------

def screen_one(instId: str, ticker: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = okx_get_candles(instId, TIMEFRAME, CANDLE_LIMIT)
    if not raw or len(raw) < 220:  # need enough history for EMA200
        return None
    cd = parse_okx_candles(raw)
    c, h, l, v = cd.close, cd.high, cd.low, cd.vol
    price = c[-1]

    # Min price cap filter (optional)
    if MIN_USDT_PRICE is not None and price >= MIN_USDT_PRICE:
        return None

    # Trend Template
    tt = trend_template_crypto(c)
    if not tt["all_ok"]:
        return None

    # Accumulation: VCP-lite or Darvas box
    vcp_ok = vcp_lite_filter(h, l, c)
    darvas_ok, box_high = darvas_box_filter(h, l, c)
    if not (vcp_ok or darvas_ok):
        return None

    # Momentum
    mom = momentum_confirm(c, h, l, v)
    if not mom["all_ok"]:
        return None

    score = momentum_score(c, h, l, v)

    # Distance to 180D high/low
    w = min(PCT_ABOVE_LOW_WINDOW_DAYS, len(c))
    win = c[-w:]
    low_w = min(win)
    high_w = max(win)
    pct_above_low = (price - low_w) / low_w if low_w > 0 else 0.0
    pct_below_high = (high_w - price) / high_w if high_w > 0 else 0.0

    # 24h volume filter (quote)
    vol_q = 0.0
    if ticker:
        # OKX provides 'volCcy24h' as base or quote depending; try both safe
        for key in ("volCcy24h", "vol24h", "volCcyQuote", "volCcy"):
            if key in ticker and ticker[key] not in (None, "", "0"):
                try:
                    vol_q = float(ticker[key])
                    break
                except:
                    pass
    # Filter by 24h quote volume (USDT) if available
    if vol_q and vol_q < MIN_24H_USDT_VOL:
        return None

    return {
        "instId": instId,
        "price": price,
        "score": score,
        "pct_above_low": pct_above_low,
        "pct_below_high": pct_below_high,
        "vcp_ok": vcp_ok,
        "darvas_ok": darvas_ok,
        "box_high": box_high,
        "vol_q_24h": vol_q,
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Fetching SPOT USDT instruments from OKX...")
    instruments = okx_get_instruments_spot_usdt()
    if not instruments:
        logging.error("No instruments fetched.")
        sys.exit(1)

    # Reduce to live USDT pairs and optionally cap count
    inst_ids = [it["instId"] for it in instruments if it.get("quoteCcy") == "USDT" and it.get("state") == "live"]
    if MAX_COINS:
        inst_ids = inst_ids[:MAX_COINS]

    logging.info(f"Total SPOT USDT pairs to scan: {len(inst_ids)}")

    tickers = okx_get_tickers_spot()

    results = []
    for i, instId in enumerate(inst_ids, 1):
        if i % 25 == 0:
            logging.info(f"Scanning {i}/{len(inst_ids)}...")
        ticker = tickers.get(instId, {})
        try:
            res = screen_one(instId, ticker)
            if res:
                results.append(res)
        except Exception as e:
            logging.warning(f"screen_one error for {instId}: {e}")

        time.sleep(0.05)  # gentle pacing

    if not results:
        msg = "🔍 Không có coin SPOT nào đạt bộ lọc Trend+VCP+Momentum ở thời điểm quét."
        logging.info(msg)
        tg_send_message(msg)
        return

    # Rank by score then by pct_below_high ascending (closer to high first)
    results.sort(key=lambda x: (-x["score"], x["pct_below_high"]))

    # Prepare Telegram message
    header = "🚀 <b>SPOT Trend Screener (OKX)</b>\n"
    header += f"Khung: <b>{TIMEFRAME}</b> | Quy tắc: Trend Template + VCP/Darvas + Momentum\n"
    header += f"Điều kiện: RSI≥{RSI_MIN}, ADX≥{ADX_MIN}, BBW≤{int(BB_WIDTH_MAX*100)}%, Vol≥{VOL_SPIKE_MULT}×MA20\n"
    lines = [header]

    for idx, r in enumerate(results[:TOP_N], 1):
        name = r["instId"].replace("-", "/")
        tags = []
        if r["vcp_ok"]: tags.append("VCP")
        if r["darvas_ok"]: tags.append("Darvas")
        tag_txt = ",".join(tags) if tags else "—"
        line = (
            f"{idx:02d}. <b>{name}</b> @ <code>{r['price']:.6g}</code> "
            f"| Score <b>{r['score']:.2f}</b> "
            f"| ↑low180 {pct(r['pct_above_low'])} | ↓high180 {pct(r['pct_below_high'])} "
            f"| {tag_txt}"
        )
        lines.append(line)

    text = "\n".join(lines)
    sent = tg_send_message(text)
    if sent:
        logging.info("Telegram message sent.")
    else:
        logging.error("Failed to send Telegram message.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
