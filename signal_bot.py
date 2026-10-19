import os, time, requests

TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]
SYMBOL = "GC=F"
NAME = "الذهب"
INTERVAL = "15m"
MANUAL = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"


def send(text):
    requests.post("https://api.telegram.org/bot" + TOKEN + "/sendMessage",
                  data={"chat_id": CHAT_ID, "text": text}, timeout=15)


def get_candles():
    url = "https://query1.finance.yahoo.com/v8/finance/chart/" + SYMBOL
    r = requests.get(url, params={"interval": INTERVAL, "range": "5d"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=15).json()
    res = r["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    rows = []
    for t, h, l, c in zip(res["timestamp"], q["high"], q["low"], q["close"]):
        if None not in (h, l, c):
            rows.append((t, h, l, c))
    return rows[:-1]


def ema(vals, span):
    k = 2 / (span + 1)
    e = vals[0]
    out = [e]
    for v in vals[1:]:
        e = v * k + e * (1 - k)
        out.append(e)
    return out


def rsi(c, n=14):
    d = [c[i] - c[i - 1] for i in range(-n, 0)]
    g = sum(x for x in d if x > 0) / n
    l = sum(-x for x in d if x < 0) / n
    return 100 if l == 0 else 100 - 100 / (1 + g / l)


def atr(rows, n=14):
    trs = []
    for i in range(-n, 0):
        _, h, l, _c = rows[i]
        pc = rows[i - 1][3]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs) / n


rows = get_candles()
closes = [r[3] for r in rows]
t, price = rows[-1][0], closes[-1]
f, s = ema(closes, 9), ema(closes, 21)

side = None
if f[-2] < s[-2] and f[-1] > s[-1] and rsi(closes) < 70:
    side = "شراء 🟢"
    sl, tp = price - 1.5 * atr(rows), price + 3 * atr(rows)
elif f[-2] > s[-2] and f[-1] < s[-1] and rsi(closes) > 30:
    side = "بيع 🔴"
    sl, tp = price + 1.5 * atr(rows), price - 3 * atr(rows)

fresh = (time.time() - (t + 900)) < 900  # الشمعة اتقفلت من أقل من 15 دقيقة
print("السعر:", round(price, 2), "| إشارة:", side, "| جديدة:", fresh)

if MANUAL:
    send("✅ البوت شغّال على GitHub\nالسعر: " + str(round(price, 2)))

if side and fresh:
    send("📢 إشارة " + side + " على " + NAME +
         "\nالدخول: " + str(round(price, 2)) +
         "\nوقف الخسارة: " + str(round(sl, 2)) +
         " (بُعد " + str(round(abs(price - sl), 2)) + ")" +
         "\nالهدف: " + str(round(tp, 2)) +
         " (بُعد " + str(round(abs(price - tp), 2)) + ")" +
         "\n⚠️ تحليل آلي مش نصيحة مالية")
