import os, json, time, math, requests
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.environ.get("CHAT_ID", "").strip()
SYMBOLS = ["XAUUSD=X", "GC=F"]   # الفوري الأول، والعقود الآجلة احتياطي
SYMBOL = SYMBOLS[0]
NAME = "الذهب"
INTERVAL = "15m"
BAR = 900            # ثواني في الشمعة
HORIZON = 48         # الصفقة تنتهي بعد 48 شمعة (12 ساعة)
MIN_PROB = 0.40      # أقل احتمال نجاح نعتبر عنده الإشارة قوية
STATE_FILE = "state.json"
MANUAL = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"


def send(text):
    if not TOKEN or not CHAT_ID:
        print("مفيش توكن:", text)
        return
    r = requests.post("https://api.telegram.org/bot" + TOKEN + "/sendMessage",
                      data={"chat_id": CHAT_ID, "text": text}, timeout=15)
    print("تليجرام:", r.status_code)


def fetch(symbol):
    url = "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol
    r = requests.get(url, params={"interval": INTERVAL, "range": "60d"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=20).json()
    res = r["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    rows = []
    for t, h, l, c in zip(res["timestamp"], q["high"], q["low"], q["close"]):
        if None not in (h, l, c):
            rows.append((t, h, l, c))
    return rows[:-1]  # نستخدم الشموع المقفولة بس


def get_candles():
    global SYMBOL
    for sym in SYMBOLS:
        try:
            rows = fetch(sym)
            if len(rows) >= 500:
                SYMBOL = sym
                return rows
            print("بيانات قليلة من", sym, len(rows))
        except Exception as e:
            print("فشل", sym, e)
    raise RuntimeError("مفيش بيانات من أي مصدر")


# ---------- المؤشرات ----------
def ema(x, span):
    k = 2 / (span + 1)
    out = np.empty(len(x))
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = x[i] * k + out[i - 1] * (1 - k)
    return out


def rolling_mean(x, n):
    out = np.full(len(x), np.nan)
    cs = np.cumsum(np.insert(x, 0, 0.0))
    out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def indicators(rows):
    t = np.array([r[0] for r in rows], dtype=float)
    h = np.array([r[1] for r in rows], dtype=float)
    l = np.array([r[2] for r in rows], dtype=float)
    c = np.array([r[3] for r in rows], dtype=float)
    f, s = ema(c, 9), ema(c, 21)
    pc = np.roll(c, 1)
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    tr[0] = h[0] - l[0]
    atr = rolling_mean(tr, 14)
    d = np.diff(c, prepend=c[0])
    gain = rolling_mean(np.clip(d, 0, None), 14)
    loss = rolling_mean(np.clip(-d, 0, None), 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = np.where(loss == 0, 100.0, 100 - 100 / (1 + gain / loss))
    return t, h, l, c, f, s, atr, rsi


def signal_at(i, f, s, rsi):
    if f[i - 1] < s[i - 1] and f[i] > s[i] and rsi[i] < 70:
        return 1
    if f[i - 1] > s[i - 1] and f[i] < s[i] and rsi[i] > 30:
        return -1
    return 0


def features(i, side, t, c, f, s, atr, rsi):
    a = atr[i]
    hour = (t[i] % 86400) / 3600
    return [
        side,
        (f[i] - s[i]) / a,
        rsi[i],
        a / c[i] * 100,
        (c[i] - c[i - 4]) / a,
        (c[i] - c[i - 16]) / a,
        (s[i] - s[i - 4]) / a,
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
    ]


def levels(side, entry, a):
    if side == 1:
        return entry - 1.5 * a, entry + 3 * a
    return entry + 1.5 * a, entry - 3 * a


def outcome(side, sl, tp, highs, lows):
    """1 = وصل الهدف، 0 = وقف الخسارة أو انتهت المدة، None = لسه"""
    for hi, lo in zip(highs, lows):
        hit_sl = lo <= sl if side == 1 else hi >= sl
        hit_tp = hi >= tp if side == 1 else lo <= tp
        if hit_sl:
            return 0          # لو الاتنين في نفس الشمعة نعتبرها خسارة
        if hit_tp:
            return 1
    if len(highs) >= HORIZON:
        return 0
    return None


# ---------- التدريب ----------
def train(rows):
    t, h, l, c, f, s, atr, rsi = indicators(rows)
    X, y, idx = [], [], []
    for i in range(30, len(rows) - 1):
        side = signal_at(i, f, s, rsi)
        if side == 0:
            continue
        sl, tp = levels(side, c[i], atr[i])
        o = outcome(side, sl, tp, h[i + 1:i + 1 + HORIZON], l[i + 1:i + 1 + HORIZON])
        if o is None:
            continue
        X.append(features(i, side, t, c, f, s, atr, rsi))
        y.append(o)
        idx.append(i)
    X, y, idx = np.array(X), np.array(y), np.array(idx)
    info = {"n": len(y), "base": float(y.mean()) if len(y) else 0.0, "test": None}
    if len(y) < 40 or len(set(y.tolist())) < 2:
        return None, info

    def make():
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=1000))

    # اختبار على آخر 30% من الإشارات (بيانات النموذج ما شافهاش)
    cut = int(len(y) * 0.7)
    train_mask = idx < idx[cut] - HORIZON
    if train_mask.sum() >= 30 and len(set(y[train_mask].tolist())) == 2:
        m = make().fit(X[train_mask], y[train_mask])
        p = m.predict_proba(X[cut:])[:, 1]
        pick = p >= MIN_PROB
        info["test"] = {
            "n": int(len(p)),
            "base": float(y[cut:].mean()),
            "picked": int(pick.sum()),
            "win": float(y[cut:][pick].mean()) if pick.sum() else None,
        }
    return make().fit(X, y), info


# ---------- الحالة والنتايج ----------
def load_state():
    try:
        with open(STATE_FILE) as fh:
            st = json.load(fh)
    except Exception:
        st = {}
    st.setdefault("open", [])
    st.setdefault("last_t", 0)
    st.setdefault("wins", 0)
    st.setdefault("losses", 0)
    st.setdefault("expired", 0)
    return st


def save_state(st):
    with open(STATE_FILE, "w") as fh:
        json.dump(st, fh, ensure_ascii=False, indent=1)


def stats_line(st):
    return ("📊 النتايج: " + str(st["wins"]) + " ربح | " +
            str(st["losses"]) + " خسارة | " + str(st["expired"]) + " انتهت")


def update_open(rows, st):
    keep = []
    for sg in st["open"]:
        later = [r for r in rows if r[0] > sg["t"]]
        res, bars = None, 0
        for r in later:
            bars += 1
            hit_sl = r[2] <= sg["sl"] if sg["side"] == 1 else r[1] >= sg["sl"]
            hit_tp = r[1] >= sg["tp"] if sg["side"] == 1 else r[2] <= sg["tp"]
            if hit_sl:
                res = "loss"
                break
            if hit_tp:
                res = "win"
                break
            if bars >= HORIZON:
                res = "expired"
                break
        word = "شراء" if sg["side"] == 1 else "بيع"
        if res == "win":
            st["wins"] += 1
            send("🎯 إشارة " + word + " (دخول " + str(sg["entry"]) + ") وصلت الهدف ✅\n" + stats_line(st))
        elif res == "loss":
            st["losses"] += 1
            send("🛑 إشارة " + word + " (دخول " + str(sg["entry"]) + ") ضربت وقف الخسارة ❌\n" + stats_line(st))
        elif res == "expired":
            st["expired"] += 1
            send("⏳ إشارة " + word + " (دخول " + str(sg["entry"]) + ") انتهت مدتها (12 ساعة) من غير هدف ولا وقف\n" + stats_line(st))
        else:
            keep.append(sg)
    st["open"] = keep


# ---------- التشغيل ----------
def main():
    rows = get_candles()
    st = load_state()
    update_open(rows, st)

    model, info = train(rows)
    t, h, l, c, f, s, atr, rsi = indicators(rows)
    i = len(rows) - 1
    price = float(c[i])
    side = signal_at(i, f, s, rsi)
    fresh = (time.time() - (t[i] + BAR)) < 3 * BAR
    print("السعر:", round(price, 2), "| إشارة:", side, "| جديدة:", fresh, "| عينة التدريب:", info["n"])

    if side != 0 and fresh and int(t[i]) > st["last_t"]:
        sl, tp = levels(side, price, atr[i])
        prob = None
        if model is not None:
            x = np.array([features(i, side, t, c, f, s, atr, rsi)])
            prob = float(model.predict_proba(x)[0][1])
        word = "شراء 🟢" if side == 1 else "بيع 🔴"
        msg = ("📢 إشارة " + word + " على " + NAME +
               "\nالدخول: " + str(round(price, 2)) +
               "\nوقف الخسارة: " + str(round(sl, 2)) + " (بُعد " + str(round(abs(price - sl), 2)) + ")" +
               "\nالهدف: " + str(round(tp, 2)) + " (بُعد " + str(round(abs(price - tp), 2)) + ")")
        if prob is not None:
            tag = "✅ قوية" if prob >= MIN_PROB else "⚠️ ضعيفة (الأفضل تتجاهليها)"
            msg += "\n🤖 تقييم الذكاء الاصطناعي: " + str(round(prob * 100)) + "% " + tag
        msg += "\n" + stats_line(st) + "\n⚠️ تحليل آلي مش نصيحة مالية"
        send(msg)
        st["open"].append({"t": int(t[i]), "side": side, "entry": round(price, 2),
                           "sl": round(float(sl), 2), "tp": round(float(tp), 2),
                           "prob": None if prob is None else round(prob, 3)})
        st["last_t"] = int(t[i])

    if MANUAL:
        msg = "✅ البوت شغّال على GitHub\nالسعر: " + str(round(price, 2)) + " (مصدر: " + SYMBOL + ")"
        msg += "\n🤖 إشارات سابقة اتدرّب عليها النموذج: " + str(info["n"])
        if info["n"]:
            msg += "\nنسبة وصول الهدف في كل الإشارات: " + str(round(info["base"] * 100)) + "%"
        te = info["test"]
        if te:
            msg += "\n🧪 اختبار على آخر " + str(te["n"]) + " إشارة: "
            if te["win"] is None:
                msg += "النموذج ما اختارش أي إشارة"
            else:
                msg += ("النموذج اختار " + str(te["picked"]) + " ونجح منهم " +
                        str(round(te["win"] * 100)) + "% (المتوسط العادي " +
                        str(round(te["base"] * 100)) + "%)")
        elif model is None:
            msg += "\n(بيانات قليلة، النموذج لسه ما اشتغلش)"
        msg += "\n" + stats_line(st)
        send(msg)

    save_state(st)


if __name__ == "__main__":
    main()
