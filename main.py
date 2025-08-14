#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OKX SPOT Trend Screener -> Telegram + Google Sheet (Service Account)
- 2 chế độ: RELAX_MODE=1 (mặc định), RELAX_MODE=0 (STRICT)
- Lọc coin SPOT/USDT giá < MAX_USDT_PRICE (mặc định 1.0 USDT)
- Ghi Google Sheet 7 cột: [Coin, Tín hiệu, Giá, Ngày, Tần suất, Type, Giá Bán dự kiến]
- DEBUG chi tiết:
  * DEBUG=0: log gọn
  * DEBUG=1: log tiến độ, coin pass từng bước, tổng kết
  * DEBUG=2: log cả lý do loại cho từng coin
  * LOG_EVERY=k: cứ k coin in 1 dòng tiến độ (mặc định 25)
"""

import os, re, time, math, logging, requests
from datetime import datetime, timezone, timedelta

# ======== ENV ========
OKX_BASE_URL = "https://www.okx.com"
TIMEFRAME = os.getenv("TIMEFRAME", "1D")
MAX_COINS = int(os.getenv("MAX_COINS", "0")) or None
MAX_USDT_PRICE = float(os.getenv("MAX_USDT_PRICE", "1.0"))
MIN_24H_USDT_VOL = float(os.getenv("MIN_24H_USDT_VOL", "1000000"))
TOP_N = int(os.getenv("TOP_N", "30"))

RELAX_MODE = int(os.getenv("RELAX_MODE", "1"))        # 1=RELAX, 0=STRICT
TYPE_LABEL = "RELAX" if RELAX_MODE else "STRICT"

# Debug controls
DEBUG = int(os.getenv("DEBUG", "1"))                   # 0/1/2
LOG_EVERY = int(os.getenv("LOG_EVERY", "25"))

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Google Sheet (Service Account)
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "/etc/secrets/service_account.json").strip()  # /etc/secrets/service_account.json
SHEET_CSV_URL = os.getenv("SHEET_CSV_URL", "").strip()                # /spreadsheets/d/<ID>/edit
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "").strip()
SHEET_NAME = os.getenv("SHEET_NAME", "DATA_SPOT")
APPEND_FREQ = int(os.getenv("APPEND_FREQ", "60"))                     # phút
TP_PCT = float(os.getenv("TP_PCT", "0.033"))                          # +3.3%

# ======== Ngưỡng lọc ========
PCT_ABOVE_LOW_WINDOW_DAYS = 180
# RELAX
PCT_ABOVE_LOW_MIN_RELAX  = 0.30
PCT_BELOW_HIGH_MAX_RELAX = 0.35
RSI_MIN_RELAX, ADX_MIN_RELAX = 50, 15
VOL_SPIKE_RELAX, BBW_MAX_RELAX = 1.05, 0.10
ATR_WIN, ATR_CONTR_LOOKBACK, ATR_NORM_MAX_RELAX = 20, 30, 0.08
DARVAS_WIN, DARVAS_W_MAX_RELAX = 15, 0.15
# STRICT
PCT_ABOVE_LOW_MIN_STRICT  = 0.30
PCT_BELOW_HIGH_MAX_STRICT = 0.25
RSI_MIN_STRICT, ADX_MIN_STRICT = 55, 20
VOL_SPIKE_STRICT, BBW_MAX_STRICT = 1.20, 0.06
ATR_NORM_MAX_STRICT, DARVAS_W_MAX_STRICT = 0.04, 0.10

CANDLE_LIMIT = 300

# ======== Helpers / Indicators ========
def ema(values, period):
    k = 2/(period+1); out, e = [], None
    for v in values:
        e = v if e is None else v*k + e*(1-k); out.append(e)
    return out

def sma(values, period):
    out, s, q = [], 0.0, []
    for v in values:
        q.append(v); s += v
        if len(q)>period: s -= q.pop(0)
        out.append(s/len(q))
    return out

def rsi(values, period=14):
    if len(values) < period+1: return [math.nan]*len(values)
    gains, losses = [], []
    for i in range(1, period+1):
        ch = values[i]-values[i-1]
        gains.append(max(ch,0)); losses.append(max(-ch,0))
    ag, al = sum(gains)/period, sum(losses)/period
    out = [math.nan]*period
    rs = (ag/al) if al!=0 else math.inf
    out.append(100 - (100/(1+rs)))
    for i in range(period+1, len(values)):
        ch = values[i]-values[i-1]; g, l = max(ch,0), max(-ch,0)
        ag = (ag*(period-1)+g)/period; al = (al*(period-1)+l)/period
        rs = (ag/al) if al!=0 else math.inf
        out.append(100 - (100/(1+rs)))
    return out

def true_range(h,l,c):
    out=[]; pc=None
    for i in range(len(c)):
        tr = (h[i]-l[i]) if pc is None else max(h[i]-l[i], abs(h[i]-pc), abs(l[i]-pc))
        out.append(tr); pc=c[i]
    return out

def atr(h,l,c,period=14): return sma(true_range(h,l,c), period)

def adx(h,l,c,period=14):
    plus_dm=[0.0]; minus_dm=[0.0]
    for i in range(1,len(h)):
        up=h[i]-h[i-1]; dn=l[i-1]-l[i]
        plus_dm.append(max(up,0.0) if up>dn and up>0 else 0.0)
        minus_dm.append(max(dn,0.0) if dn>up and dn>0 else 0.0)
    atr_v = atr(h,l,c,period); pd = sma(plus_dm, period); md = sma(minus_dm, period)
    plus_di=[]; minus_di=[]
    for i in range(len(c)):
        if atr_v[i] and atr_v[i]!=0:
            plus_di.append(100*(pd[i]/atr_v[i])); minus_di.append(100*(md[i]/atr_v[i]))
        else:
            plus_di.append(math.nan); minus_di.append(math.nan)
    dx=[]
    for i in range(len(c)):
        a,b=plus_di[i],minus_di[i]
        dx.append(math.nan if (math.isnan(a) or math.isnan(b) or (a+b)==0) else 100*abs(a-b)/(a+b))
    return sma(dx, period), plus_di, minus_di

def bbands(values, period=20, std_mult=2.0):
    ma = sma(values, period)
    import statistics
    upper, lower, width = [], [], []
    for i in range(len(values)):
        if i<period:
            upper.append(math.nan); lower.append(math.nan); width.append(math.nan)
        else:
            sl = values[i-period+1:i+1]; sd = statistics.pstdev(sl)
            u = ma[i]+std_mult*sd; d = ma[i]-std_mult*sd
            upper.append(u); lower.append(d)
            width.append((u-d)/values[i] if values[i]!=0 else math.nan)
    return ma, upper, lower, width

# ======== OKX API ========
def okx_get_instruments():
    r = requests.get(f"{OKX_BASE_URL}/api/v5/public/instruments", params={"instType":"SPOT"}, timeout=20)
    j = r.json(); return [x for x in j.get("data",[]) if x.get("quoteCcy")=="USDT" and x.get("state")=="live"]

def okx_get_tickers():
    r = requests.get(f"{OKX_BASE_URL}/api/v5/market/tickers", params={"instType":"SPOT"}, timeout=20)
    j = r.json(); return {x["instId"]:x for x in j.get("data",[])}

def okx_get_candles(instId, bar, limit=300):
    r = requests.get(f"{OKX_BASE_URL}/api/v5/market/candles", params={"instId":instId,"bar":bar,"limit":limit}, timeout=20)
    j = r.json(); return list(reversed(j.get("data",[])))  # oldest-first

# ======== Trend/Momentum Filters ========
def trend_template_relax(c):
    ema50, ema150, ema200 = ema(c,50), ema(c,150), ema(c,200)
    price = c[-1]
    if not (price>ema50[-1]>ema150[-1]>ema200[-1]): return False
    w=min(PCT_ABOVE_LOW_WINDOW_DAYS,len(c)); low=min(c[-w:]); high=max(c[-w:])
    if low<=0 or high<=0: return False
    if (price-low)/low < PCT_ABOVE_LOW_MIN_RELAX: return False
    if (high-price)/high > PCT_BELOW_HIGH_MAX_RELAX: return False
    return True

def trend_template_strict(c):
    ema50, ema150, ema200 = ema(c,50), ema(c,150), ema(c,200)
    price = c[-1]
    if not (price>ema50[-1] and price>ema150[-1] and price>ema200[-1]): return False
    if not (ema50[-1]>ema150[-1]>ema200[-1]): return False
    if len(ema200)>5 and (ema200[-1]-ema200[-5])<=0: return False  # EMA200 phải dốc lên
    w=min(PCT_ABOVE_LOW_WINDOW_DAYS,len(c)); low=min(c[-w:]); high=max(c[-w:])
    if low<=0 or high<=0: return False
    if (price-low)/low < PCT_ABOVE_LOW_MIN_STRICT: return False
    if (high-price)/high > PCT_BELOW_HIGH_MAX_STRICT: return False
    return True

def vcp_lite(h,l,c, atr_norm_max):
    a = atr(h,l,c, ATR_WIN)
    if len(a)<ATR_CONTR_LOOKBACK+5: return False
    recent = a[-ATR_CONTR_LOOKBACK:]
    if any(math.isnan(x) for x in recent): return False
    slope = recent[-1] - recent[0]
    norm = recent[-1]/c[-1] if c[-1]>0 else 1.0
    return (slope<0) and (norm <= atr_norm_max)

def darvas_box(h,l,c, max_width):
    if len(c)<DARVAS_WIN+2: return False
    bh, bl = max(h[-DARVAS_WIN:]), min(l[-DARVAS_WIN:])
    width = (bh-bl)/c[-1] if c[-1]>0 else 1.0
    broke = c[-1] > bh*1.005  # +0.5% buffer
    return broke and (width<=max_width)

def momentum(c,h,l,v, *, strict=False):
    rsi_v = rsi(c,14)[-1]
    adx_v, pdi, mdi = adx(h,l,c,14)
    adx_last = adx_v[-1]
    _,_,_, bbw = bbands(c,20,2.0)
    vol_ma20 = sma(v,20)
    if strict:
        return (rsi_v>=RSI_MIN_STRICT and adx_last>=ADX_MIN_STRICT and
                (pdi[-1]>mdi[-1]) and
                (vol_ma20[-1]>0 and v[-1]>=vol_ma20[-1]*VOL_SPIKE_STRICT) and
                (not math.isnan(bbw[-1]) and bbw[-1]<=BBW_MAX_STRICT))
    else:
        conds = [
            rsi_v>=RSI_MIN_RELAX,
            adx_last>=ADX_MIN_RELAX,
            (pdi[-1]>mdi[-1]),
            (vol_ma20[-1]>0 and v[-1]>=vol_ma20[-1]*VOL_SPIKE_RELAX),
            (not math.isnan(bbw[-1]) and bbw[-1]<=BBW_MAX_RELAX)
        ]
        return sum(conds)>=3

# ======== Telegram ========
def tg_send(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.error("Missing TELEGRAM_BOT_TOKEN/CHAT_ID"); return False
    url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload={"chat_id":TELEGRAM_CHAT_ID,"text":text,"parse_mode":"HTML","disable_web_page_preview":True}
    try:
        r=requests.post(url,json=payload,timeout=20)
        return r.status_code==200
    except Exception as e:
        logging.error(f"Telegram exception: {e}")
        return False

# ======== Google Sheet (Service Account) ========
import gspread
from google.oauth2.service_account import Credentials

def extract_spreadsheet_id(sheet_csv_url: str) -> str:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)/", sheet_csv_url or "")
    return m.group(1) if m else ""

if not SPREADSHEET_ID and SHEET_CSV_URL:
    SPREADSHEET_ID = extract_spreadsheet_id(SHEET_CSV_URL)

def sheet_append_rows_service(rows):
    if not SERVICE_ACCOUNT_FILE:
        logging.warning("SERVICE_ACCOUNT_FILE chưa cấu hình."); return False
    if not SPREADSHEET_ID:
        logging.warning("SPREADSHEET_ID rỗng (không rút được từ SHEET_CSV_URL)."); return False
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    try:
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SPREADSHEET_ID)
        try:
            ws = sh.worksheet(SHEET_NAME)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=SHEET_NAME, rows=1000, cols=20)
        ws.append_rows(rows, value_input_option="USER_ENTERED")
        return True
    except Exception as e:
        logging.error(f"[Sheet] append lỗi: {e}")
        return False

# ======== Screening (with debug) ========
def screen_market():
    insts = okx_get_instruments()
    total_pairs = len(insts)
    if MAX_COINS: insts = insts[:MAX_COINS]
    tickers = okx_get_tickers()

    logging.info(f"Pairs: {total_pairs} | Mode: {TYPE_LABEL} | TF: {TIMEFRAME} | Scan: {len(insts)}")

    # Counters
    cnt_seen = cnt_price = cnt_vol = cnt_trend = cnt_mom = 0
    cnt_vcp = cnt_darvas = cnt_strict_final = 0

    results=[]
    for i,it in enumerate(insts, 1):
        instId = it["instId"]
        tk = tickers.get(instId, {})

        # progress
        if i % LOG_EVERY == 0 or DEBUG >= 1:
            logging.info(f"Scanning {i}/{len(insts)} ... {instId}")

        reason = None
        cnt_seen += 1

        # Price filter
        try: price = float(tk.get("last","0") or 0)
        except: price = 0.0
        if price<=0 or price>=MAX_USDT_PRICE:
            reason = f"price_filter (price={price})"
            if DEBUG>=2: logging.info(f"❌ {instId} bị loại: {reason}")
            continue
        cnt_price += 1

        # Volume quote 24h
        vol_q=0.0
        for key in ("volCcy24h","vol24h","volCcyQuote","volCcy"):
            if key in tk and tk[key] not in (None,"","0"):
                try: vol_q=float(tk[key]); break
                except: pass
        if vol_q < MIN_24H_USDT_VOL:
            reason = f"vol24h_quote<{MIN_24H_USDT_VOL} (got {vol_q})"
            if DEBUG>=2: logging.info(f"❌ {instId} bị loại: {reason}")
            continue
        cnt_vol += 1

        # Candles
        raw = okx_get_candles(instId, TIMEFRAME, CANDLE_LIMIT)
        if not raw or len(raw)<220:
            reason = "not_enough_candles"
            if DEBUG>=2: logging.info(f"❌ {instId} bị loại: {reason}")
            continue
        h,l,c,v=[],[],[],[]
        for row in raw:
            h.append(float(row[2])); l.append(float(row[3])); c.append(float(row[4])); v.append(float(row[5]))

        # Trend + Momentum (+VCP/Darvas if STRICT)
        if RELAX_MODE:
            if not trend_template_relax(c):
                reason = "trend_relax_fail"
                if DEBUG>=2: logging.info(f"❌ {instId} bị loại: {reason}")
                continue
            cnt_trend += 1

            if not momentum(c,h,l,v, strict=False):
                reason = "momentum_relax_fail"
                if DEBUG>=2: logging.info(f"❌ {instId} bị loại: {reason}")
                continue
            cnt_mom += 1

            results.append({"instId":instId, "price":price})
            if DEBUG>=1: logging.info(f"✅ PASS (RELAX): {instId} @ {price:.6g}")

        else:
            if not trend_template_strict(c):
                reason = "trend_strict_fail"
                if DEBUG>=2: logging.info(f"❌ {instId} bị loại: {reason}")
                continue
            cnt_trend += 1

            mom_ok = momentum(c,h,l,v, strict=True)
            vcp_ok = vcp_lite(h,l,c, ATR_NORM_MAX_STRICT)
            darvas_ok = darvas_box(h,l,c, DARVAS_W_MAX_STRICT)
            if mom_ok: cnt_mom += 1
            if vcp_ok: cnt_vcp += 1
            if darvas_ok: cnt_darvas += 1

            if not (mom_ok and (vcp_ok or darvas_ok)):
                reason = f"strict_combo_fail (mom={mom_ok}, vcp={vcp_ok}, darvas={darvas_ok})"
                if DEBUG>=2: logging.info(f"❌ {instId} bị loại: {reason}")
                continue

            cnt_strict_final += 1
            results.append({"instId":instId, "price":price})
            if DEBUG>=1: logging.info(f"✅ PASS (STRICT): {instId} @ {price:.6g}")

        time.sleep(0.04)  # nhẹ nhàng API

    # Summary counters
    logging.info(
        f"Summary: seen={cnt_seen}, price_ok={cnt_price}, vol_ok={cnt_vol}, "
        f"trend_ok={cnt_trend}, mom_ok={cnt_mom}, "
        f"vcp_ok={cnt_vcp}, darvas_ok={cnt_darvas}, final={len(results)}"
    )

    return results

# ======== Orchestrate ========
def run_once():
    results = screen_market()

    # ===== Telegram =====
    sent = False; sent_count = 0
    if not results:
        sent = tg_send(f"Không có coin SPOT <{MAX_USDT_PRICE}$ đạt lọc {TYPE_LABEL}.")
        sent_count = 0
        logging.info(f"Telegram sent_empty={sent}")
    else:
        head = f"🚀 <b>SPOT Trend {TYPE_LABEL} (&lt;{MAX_USDT_PRICE}$)</b>\n"
        lines = [f"{i:02d}. {r['instId']} @ {r['price']:.6g}" for i,r in enumerate(results[:TOP_N],1)]
        sent = tg_send(head + "\n".join(lines))
        sent_count = min(len(results), TOP_N)
        logging.info(f"Telegram sent={sent} | rows={sent_count}")

    # ===== Google Sheet (7 cột) =====
    sheet_ok = False; sheet_rows = 0
    if results and SERVICE_ACCOUNT_FILE:
        now_vn = datetime.now(timezone.utc) + timedelta(hours=7)
        now_str = now_vn.strftime("%Y-%m-%d %H:%M:%S")
        rows=[]
        for r in results[:TOP_N]:
            coin = r["instId"]; price = r["price"]; tp_price = price*(1.0+TP_PCT)
            rows.append([coin, "MUA MẠNH", price, now_str, APPEND_FREQ, TYPE_LABEL, round(tp_price, 10)])
        sheet_ok = sheet_append_rows_service(rows)
        sheet_rows = len(rows) if sheet_ok else 0
        logging.info(f"Sheet append ok={sheet_ok} | rows={sheet_rows} | sheet={SHEET_NAME}")
    elif results and not SERVICE_ACCOUNT_FILE:
        logging.warning("Có kết quả nhưng thiếu SERVICE_ACCOUNT_FILE -> không ghi sheet.")

    # Final line for quick glance
    logging.info(f"DONE | found={len(results)} | sent_tg={sent_count} | sheet_rows={sheet_rows} | mode={TYPE_LABEL} | TF={TIMEFRAME}")

# ======== Entry ========
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        run_once()
    except KeyboardInterrupt:
        pass
        
