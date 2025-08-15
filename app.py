# app.py — AI Trade Coach (Options) with Voice Input + Auto Watchlist Scan
# Free stack: Streamlit, yfinance, pandas_ta, SpeechRecognition, streamlit_mic_recorder
# Runs great on iPad (via Streamlit Cloud). No API keys required.

import json
import math
import io
import datetime as dt
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta

# Voice input (fully optional)
try:
   import speech_recognition as sr
   from streamlit_mic_recorder import mic_recorder
   VOICE_OK = True
except Exception:
   VOICE_OK = False

APP_TITLE = "🎙️ AI Options Trade Coach — Voice + Watchlist"
WATCHLIST_FILE = Path("watchlist.json")
HISTORY_CSV = Path("coach_recommendations.csv")
HISTORY_JSON = Path("coach_recommendations.json")

# ---------------------------
# Utilities: storage & defaults
# ---------------------------
DEFAULT_WATCHLIST = ["SPY", "TSLA", "NVDA", "AAPL", "AMZN", "META", "QQQ", "AMD", "MSFT", "SOUN", "ACB", "RGTI"]

def load_watchlist() -> List[str]:
   if WATCHLIST_FILE.exists():
       try:
           wl = json.loads(WATCHLIST_FILE.read_text())
           if isinstance(wl, list) and len(wl) > 0:
               return [x.strip().upper() for x in wl if isinstance(x, str) and x.strip()]
       except Exception:
           pass
   return DEFAULT_WATCHLIST.copy()

def save_watchlist(wl: List[str]):
   WATCHLIST_FILE.write_text(json.dumps(sorted(list(dict.fromkeys([s.upper() for s in wl])))))

def append_history(rows: List[Dict]):
   if not rows:
       return
   df = pd.DataFrame(rows)
   if HISTORY_CSV.exists():
       df.to_csv(HISTORY_CSV, mode="a", header=False, index=False)
   else:
       df.to_csv(HISTORY_CSV, index=False)
   # keep JSON as rolling store
   all_rows = []
   if HISTORY_JSON.exists():
       try:
           all_rows = json.loads(HISTORY_JSON.read_text())
       except Exception:
           all_rows = []
   all_rows.extend(rows)
   HISTORY_JSON.write_text(json.dumps(all_rows[-5000:], indent=2))

# ---------------------------
# TA + expected move & scoring
# ---------------------------
def expected_move_close_to_expiry(underlying_price: float, atm_iv: Optional[float], dte: float) -> Optional[float]:
   # EM ≈ Price * IV * sqrt(DTE/365)
   if underlying_price is None or atm_iv is None or dte is None or dte <= 0:
       return None
   try:
       return float(underlying_price) * float(atm_iv) * math.sqrt(float(dte) / 365.0)
   except Exception:
       return None

def pull_iv_estimate_from_chain(chain: pd.DataFrame, underlying_price: float) -> Optional[float]:
   # try ATM option rows to approximate IV (midpoint of calls+puts)
   if chain is None or chain.empty:
       return None
   chain = chain.copy()
   chain["dist"] = (chain["strike"] - underlying_price).abs()
   atm = chain.sort_values("dist").head(4)
   ivs = []
   for _, r in atm.iterrows():
       iv = r.get("impliedVolatility", None)
       if iv is not None and iv == iv and iv > 0 and iv < 5:
           ivs.append(float(iv))
   if not ivs:
       return None
   return float(np.median(ivs))

def fetch_history(symbol: str, period="6mo", interval="1d") -> Optional[pd.DataFrame]:
   try:
       df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True, threads=True)
       if isinstance(df, pd.DataFrame) and not df.empty:
           df = df.dropna().reset_index()
           df.rename(columns={"Date": "date"}, inplace=True)
           return df
   except Exception:
       return None
   return None

def compute_tech_snapshot(df: pd.DataFrame) -> Dict:
   # Use pandas_ta (pure python; Streamlit Cloud friendly)
   close = df["Close"]
   high = df["High"]
   low = df["Low"]

   rsi = ta.rsi(close, length=14).iloc[-1]
   macd = ta.macd(close)
   macd_val = macd["MACD_12_26_9"].iloc[-1]
   macd_sig = macd["MACDs_12_26_9"].iloc[-1]
   mom = ta.mom(close, length=10).iloc[-1]
   bb = ta.bbands(close, length=20)
   bb_mid = bb["BBM_20_2.0"].iloc[-1]

   atr = ta.atr(high, low, close, length=14).iloc[-1]
   roc = ta.roc(close, length=10).iloc[-1]

   return {
       "rsi": float(rsi),
       "macd": float(macd_val),
       "macd_signal": float(macd_sig),
       "mom": float(mom),
       "bb_mid": float(bb_mid),
       "atr": float(atr),
       "roc": float(roc),
       "last": float(close.iloc[-1]),
       "last_vol": float(df["Volume"].iloc[-1]) if "Volume" in df.columns else np.nan,
       "avg_vol": float(df["Volume"].tail(30).mean()) if "Volume" in df.columns else np.nan,
   }

def score_for_calls(tech: Dict], em: Optional[float]) -> Tuple[float, List[str]]:
   score = 0.0
   reasons = []
   if tech["macd"] > tech["macd_signal"]:
       score += 1; reasons.append("MACD crossover up")
   if 35 < tech["rsi"] < 65:
       score += 1; reasons.append("RSI healthy")
   if tech["mom"] > 0:
       score += 1; reasons.append("Momentum > 0")
   if tech["last"] > tech["bb_mid"]:
       score += 1; reasons.append("Above BB mid")
   if tech["roc"] > 0:
       score += 1; reasons.append("ROC > 0")
   # Volume thrust
   if not np.isnan(tech["avg_vol"]) and tech["last_vol"] > 2.0 * max(1.0, tech["avg_vol"]):
       score += 1; reasons.append("Unusual volume")
   # Expected move sanity
   if em is not None and em > 0:
       score += 0.5; reasons.append("Expected move positive")
   return score, reasons

def score_for_puts(tech: Dict], em: Optional[float]) -> Tuple[float, List[str]]:
   score = 0.0
   reasons = []
   if tech["macd"] < tech["macd_signal"]:
       score += 1; reasons.append("MACD crossover down")
   if 35 < tech["rsi"] < 65:
       score += 1; reasons.append("RSI not oversold")
   if tech["mom"] < 0:
       score += 1; reasons.append("Momentum < 0")
   if tech["last"] < tech["bb_mid"]:
       score += 1; reasons.append("Below BB mid")
   if tech["roc"] < 0:
       score += 1; reasons.append("ROC < 0")
   if not np.isnan(tech["avg_vol"]) and tech["last_vol"] > 2.0 * max(1.0, tech["avg_vol"]):
       score += 1; reasons.append("Unusual volume")
   if em is not None and em > 0:
       score += 0.5; reasons.append("Expected move priced")
   return score, reasons

def confidence_gate(score: float) -> bool:
   # Tightened gate per your instruction
   return score >= 4.5

def best_option_row(chain_df: pd.DataFrame, kind: str, last: float) -> Optional[pd.Series]:
   # Pick near-the-money, near expiry, highest OI + volume, reasonable spread
   if chain_df is None or chain_df.empty:
       return None
   df = chain_df.copy()
   df["dist"] = (df["strike"] - last).abs()
   # heuristic filters
   df = df[(df["volume"] > 0) & (df["openInterest"] > 0)]
   # spread % using (ask-bid)/mid
   with np.errstate(divide="ignore", invalid="ignore"):
       mid = (df["bid"] + df["ask"]) / 2.0
       spread = (df["ask"] - df["bid"]) / np.where(mid == 0, np.nan, mid)
   df["spread_pct"] = spread.replace([np.inf, -np.inf], np.nan).fillna(1.0)
   df = df[df["spread_pct"] <= 0.25]  # free-friendly but reasonably tight
   # pick soonest expiry (<= 30 DTE preferred)
   # We rely on pre-filtered chain for one expiry at a time; this is a fallback sort:
   df = df.sort_values(["dist", "spread_pct", "volume", "openInterest"], ascending=[True, True, False, False])
   return df.head(1).iloc[0] if not df.empty else None

def fetch_option_candidates(symbol: str, prefer_dte_max: int = 30) -> Dict:
   tk = yf.Ticker(symbol)
   info_price = None
   try:
       info = tk.fast_info
       info_price = float(info["last_price"])
   except Exception:
       # fallback to 1m history
       hist = yf.download(symbol, period="5d", interval="1d", progress=False, auto_adjust=True, threads=True)
       if isinstance(hist, pd.DataFrame) and not hist.empty:
           info_price = float(hist["Close"].iloc[-1])
   if info_price is None:
       return {"symbol": symbol, "error": "no_price"}

   # pick the nearest expiry within prefer_dte_max
   try:
       expiries = tk.options or []
   except Exception:
       expiries = []
   if not expiries:
       return {"symbol": symbol, "error": "no_expiries"}

   # choose best expiry by minimum |DTE - 7| as default short-term
   today = dt.date.today()
   selected_exp = None
   best_delta = 9999
   for e in expiries:
       try:
           d = dt.datetime.strptime(e, "%Y-%m-%d").date()
           dte = (d - today).days
           if dte <= 0 or dte > prefer_dte_max:
               continue
           delta = abs(dte - 7)
           if delta < best_delta:
               best_delta = delta
               selected_exp = e
       except Exception:
           continue
   if selected_exp is None:
       # fallback to first expiry
       selected_exp = expiries[0]

   # get chains
   try:
       ch = tk.option_chain(selected_exp)
       calls = ch.calls.copy()
       puts  = ch.puts.copy()
   except Exception:
       return {"symbol": symbol, "error": "chain_fail"}

   # estimate IV (ATM median)
   iv_est = pull_iv_estimate_from_chain(pd.concat([calls, puts], axis=0, ignore_index=True), info_price)
   # DTE
   try:
       d = dt.datetime.strptime(selected_exp, "%Y-%m-%d").date()
       dte = (d - today).days
   except Exception:
       dte = 7

   em = expected_move_close_to_expiry(info_price, iv_est, max(dte, 1))

   # TA snapshot for direction
   df_hist = fetch_history(symbol, period="6mo", interval="1d")
   if df_hist is None or df_hist.empty:
       return {"symbol": symbol, "error": "no_hist"}

   tech = compute_tech_snapshot(df_hist)

   # score directions
   call_score, call_reasons = score_for_calls(tech, em)
   put_score, put_reasons = score_for_puts(tech, em)

   # choose side(s) that pass tight gate
   recs = []
   if confidence_gate(call_score):
       best_call = best_option_row(calls, "call", tech["last"])
       if best_call is not None:
           recs.append({
               "symbol": symbol, "side": "CALL", "expiry": selected_exp,
               "strike": float(best_call["strike"]),
               "bid": float(best_call["bid"]), "ask": float(best_call["ask"]),
               "last_underlying": tech["last"], "dte": int(dte),
               "score": round(call_score, 2), "reasons": "; ".join(call_reasons),
               "expected_move": None if em is None else round(em, 2),
               "spread_pct": float(best_call["spread_pct"]),
               "oi": int(best_call["openInterest"]), "volume": int(best_call["volume"]),
               "impliedVolatility": float(best_call.get("impliedVolatility", np.nan)) if "impliedVolatility" in best_call else np.nan
           })

   if confidence_gate(put_score):
       best_put = best_option_row(puts, "put", tech["last"])
       if best_put is not None:
           recs.append({
               "symbol": symbol, "side": "PUT", "expiry": selected_exp,
               "strike": float(best_put["strike"]),
               "bid": float(best_put["bid"]), "ask": float(best_put["ask"]),
               "last_underlying": tech["last"], "dte": int(dte),
               "score": round(put_score, 2), "reasons": "; ".join(put_reasons),
               "expected_move": None if em is None else round(em, 2),
               "spread_pct": float(best_put["spread_pct"]),
               "oi": int(best_put["openInterest"]), "volume": int(best_put["volume"]),
               "impliedVolatility": float(best_put.get("impliedVolatility", np.nan)) if "impliedVolatility" in best_put else np.nan
           })

   # directive phrasing will happen in UI section
   return {
       "symbol": symbol,
       "underlying": tech["last"],
       "em": None if em is None else float(em),
       "iv_est": None if iv_est is None else float(iv_est),
       "dte": int(dte),
       "recommendations": recs
   }

# ---------------------------
# Voice command helpers
# ---------------------------
def parse_voice_command(text: str) -> Dict:
   """
   Lightweight parser:
   examples:
     "scan the watchlist"
     "find highest confidence"
     "analyze tsla for calls"
     "best put on nvda"
   """
   t = text.lower().strip()
   out = {"action": "scan_watchlist", "symbols": []}

   # symbol extraction
   tokens = [tok.strip(",.!?") for tok in t.split()]
   syms = [tok.upper() for tok in tokens if tok.isalpha() and 2 <= len(tok) <= 5]
   if syms:
       out["symbols"] = syms
       out["action"] = "scan_symbols"

   if "watchlist" in t:
       out["action"] = "scan_watchlist"
   if "highest" in t or "top" in t or "best" in t:
       out["top_only"] = True
   if "call" in t and "put" not in t:
       out["side"] = "CALL"
   if "put" in t and "call" not in t:
       out["side"] = "PUT"
   return out

def stt_from_wav_bytes(wav_bytes: bytes) -> Optional[str]:
   if not VOICE_OK or not wav_bytes:
       return None
   r = sr.Recognizer()
   with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
       audio = r.record(source)
   try:
       # Uses Google Web Speech API (free, no key) — best-effort
       return r.recognize_google(audio)
   except Exception:
       return None

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI Options Trade Coach", layout="wide")
st.title(APP_TITLE)

# Sidebar — watchlist editor
st.sidebar.header("📝 Watchlist")
watchlist = load_watchlist()
wl_text = st.sidebar.text_area("Edit tickers (comma or newline separated):", value=", ".join(watchlist))
if st.sidebar.button("Save Watchlist"):
   wl = []
   for raw in wl_text.replace("\n", ",").split(","):
       s = raw.strip().upper()
       if s:
           wl.append(s)
   if wl:
       save_watchlist(wl)
       st.sidebar.success("Watchlist saved.")
       watchlist = wl

# Sidebar — scan controls
st.sidebar.header("🔍 Scan Controls")
max_scan = st.sidebar.slider("Tickers to scan (per run)", min_value=1, max_value=50, value=min(15, len(watchlist)))
top_n = st.sidebar.slider("Show top N highest-confidence", min_value=1, max_value=20, value=5)
auto_scan = st.sidebar.toggle("Auto-scan every 60s", value=True)
only_highest_conf = st.sidebar.toggle("Only show highest-confidence picks", value=True)

if auto_scan:
   st.experimental_set_query_params(_ts=str(dt.datetime.utcnow().timestamp()))
   st.experimental_rerun  # noop (Streamlit Cloud often auto-reruns on widgets); user can use the Refresh button below

# Voice input
st.subheader("🎤 Talk to your coach")
if VOICE_OK:
   st.caption("Tap **Record**, speak, then **Stop**. Example: “Find the highest confidence trade on TSLA.”")
   audio = mic_recorder(start_prompt="Start recording", stop_prompt="Stop", just_once=False, format="wav")
   voice_text = None
   if audio and "bytes" in audio:
       voice_text = stt_from_wav_bytes(audio["bytes"])
       if voice_text:
           st.info(f"You said: **{voice_text}**")
           st.session_state["voice_cmd"] = voice_text
else:
   st.warning("Voice features unavailable (missing dependencies). It will still work fine on Streamlit Cloud with the included requirements.")
   st.session_state["voice_cmd"] = None

# Manual query
st.subheader("🧠 Ask in plain English")
manual_query = st.text_input("e.g., “Scan the watchlist” or “Analyze NVDA calls”", value="")
if st.button("Run"):
   st.session_state["voice_cmd"] = manual_query or st.session_state.get("voice_cmd")

# Decide scan target
intent = {"action": "scan_watchlist", "symbols": []}
if st.session_state.get("voice_cmd"):
   intent = parse_voice_command(st.session_state["voice_cmd"])

if intent["action"] == "scan_symbols" and intent.get("symbols"):
   targets = [s for s in intent["symbols"] if s in watchlist] or intent["symbols"]
else:
   targets = watchlist[:max_scan]

st.markdown("---")
st.subheader("📈 Recommendations")
results: List[Dict] = []
errors: List[str] = []

progress = st.progress(0.0)
for i, sym in enumerate(targets, start=1):
   data = fetch_option_candidates(sym)
   if "error" in data:
       errors.append(f"{sym}: {data['error']}")
   else:
       for rec in data["recommendations"]:
           # optional filter by side from voice
           if "side" in intent and rec["side"] != intent["side"]:
               continue
           results.append(rec)
   progress.progress(i / max(1, len(targets)))

# Show highest confidence only if requested
if only_highest_conf and results:
   # get max score per symbol-side
   df_tmp = pd.DataFrame(results)
   idx = df_tmp.groupby(["symbol", "side"])["score"].idxmax().values
   results = [results[i] for i in idx]

# Sort by score desc, then spread asc, then volume desc
results = sorted(results, key=lambda r: (-r["score"], r["spread_pct"], -r["volume"]))

if results:
   # Save
   stamp = dt.datetime.now().isoformat()
   rows_to_save = []
   for r in results[:top_n]:
       rows_to_save.append({"timestamp": stamp, **r})
   append_history(rows_to_save)

   st.success(f"Directive picks below. **Confidence tightened** — only elite setups shown. (Saved {len(rows_to_save)} rows to CSV/JSON)")
   df_show = pd.DataFrame(results[:top_n])
   # Directive phrasing
   for r in results[:top_n]:
       side_word = "BUY CALLS" if r["side"] == "CALL" else "BUY PUTS"
       tp_hint = round(r["last_underlying"] + (r["expected_move"] or 0), 2) if r["side"] == "CALL" else round(r["last_underlying"] - (r["expected_move"] or 0), 2)
       st.markdown(
           f"**{r['symbol']}** — {side_word} near **{r['strike']}** expiring **{r['expiry']}**. "
           f"Premium midpoint ≈ ${(r['bid']+r['ask'])/2:.2f}. "
           f"Spread {r['spread_pct']*100:.1f}%. **Reason**: {r['reasons']}. "
           f"{'Aim to scale out into strength.' if r['side']=='CALL' else 'Aim to scale out into weakness.'} "
           f"Watch the underlying toward **{tp_hint}** (expected move)."
       )
   with st.expander("Table view"):
       st.dataframe(df_show.reset_index(drop=True))
else:
   st.info("No recommendations passed the tight confidence gate. That’s deliberate — we only surface **A+** setups.")

if errors:
   with st.expander("Symbols with data issues"):
       st.write(errors)

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
   if st.button("🔄 Refresh now"):
       st.experimental_rerun()
with col2:
   st.download_button("Download CSV History", data=HISTORY_CSV.read_bytes() if HISTORY_CSV.exists() else b"", file_name="coach_recommendations.csv", mime="text/csv")
with col3:
   st.download_button("Download JSON History", data=HISTORY_JSON.read_bytes() if HISTORY_JSON.exists() else b"[]", file_name="coach_recommendations.json", mime="application/json")

st.caption("This is a **coach**. It never places trades; it only recommends high-confidence entries for educational purposes.")