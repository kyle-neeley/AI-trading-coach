# 🎙️ AI Options Trade Coach — Voice + Watchlist (Free)

Zero-cost, Streamlit-based coach that:
- Scans an editable watchlist
- Calculates expected move (from yfinance IV approximation)
- Uses robust technical scoring (pandas_ta)
- Surfaces **only highest-confidence** CALL/PUT ideas with directive guidance
- Stores results to CSV + JSON
- Optional **voice input** (iPad-friendly)

## 🚀 One-Click Deploy (Streamlit Cloud)
1. Create a new GitHub repo and add these files:
  - `app.py`
  - `requirements.txt`
  - `watchlist.json`
  - `README.md`
2. Go to [share.streamlit.io](https://share.streamlit.io), connect repo, pick `app.py`, deploy.
3. Open the app on your iPad and go.

## 🗣️ Voice Input
- Uses `streamlit-mic-recorder` + `SpeechRecognition` (Google Web Speech, no key).
- If voice fails or you prefer not to use it, use the text box. The rest of the app works the same.

## 🧠 What “Highest-Confidence” Means
- Tightened scoring gate (min score 4.5) combining:
 - MACD direction, RSI health, momentum, BB mid location, ROC trend
 - Unusual volume vs 30-day average
 - Expected move sanity using IV and DTE
- Filters option contracts for:
 - Near-the-money strikes, near-term expiries (≈ 1 week)
 - Reasonable spreads (≤ 25%), non-zero volume & OI

## 📁 Data & Storage
- Watchlist lives in `watchlist.json` (editable inside the app)
- Recommendations saved to `coach_recommendations.csv` and `.json`
- Download buttons are in the UI

## 💡 Notes
- This is a **coach** — it never places trades for you.
- yfinance/IV approximations are best-effort and free; real-time accuracy varies.
- For best responsiveness, keep watchlist under ~30 tickers.
