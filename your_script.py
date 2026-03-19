import pandas as pd
import numpy as np
import requests
import json
import io
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import pytz

TW_TZ = pytz.timezone('Asia/Taipei')

# ─────────────────────────────────────────
# 1. 資料來源：官方 CSV 下載（最穩定）
# ─────────────────────────────────────────
def fetch_official_csv():
    """
    台灣彩券官方每年 CSV，每月 5 日更新至前一個月。
    民國年：2025 = 114年、2026 = 115年
    """
    now = datetime.now(TW_TZ)
    roc_year = now.year - 1911          # 民國年
    all_results = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Referer': 'https://www.taiwanlottery.com/'
    }

    # 抓今年 + 去年（確保資料足夠）
    for y in [roc_year, roc_year - 1]:
        url = f"https://www.taiwanlottery.com/lotto/history/result_download/download?year={y}"
        try:
            res = requests.get(url, headers=headers, timeout=30)
            if res.status_code != 200:
                print(f"⚠️  {y}年 CSV 回傳 {res.status_code}，跳過")
                continue

            # 偵測編碼（官方通常是 big5 或 utf-8-sig）
            for enc in ['utf-8-sig', 'big5', 'utf-8']:
                try:
                    text = res.content.decode(enc)
                    break
                except Exception:
                    continue

            df = pd.read_csv(io.StringIO(text))
            # 欄位：遊戲名稱,期別,開獎日期,...,獎號1,獎號2,獎號3,獎號4,獎號5,獎號6,特別號
            # 只取大樂透
            mask = df.iloc[:, 0].astype(str).str.contains('大樂透', na=False)
            lotto_df = df[mask]

            for _, row in lotto_df.iterrows():
                try:
                    # 找獎號欄（第6~12欄，依官方格式）
                    nums = [int(row.iloc[i]) for i in range(6, 13)
                            if str(row.iloc[i]).strip().isdigit()]
                    if len(nums) == 7:
                        all_results.append(nums)
                except Exception:
                    continue

            print(f"✅ {y}年 CSV：解析到 {len(lotto_df)} 筆大樂透")

        except Exception as e:
            print(f"❌ {y}年 CSV 失敗: {e}")

    return all_results


# ─────────────────────────────────────────
# 2. 備援：爬新版歷史開獎頁（JSON API）
# ─────────────────────────────────────────
def fetch_history_page_backup():
    """
    爬新版網址 /lotto/history/history_result/
    嘗試抓 XHR / JSON endpoint
    """
    all_results = []
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'application/json, text/javascript, */*',
        'X-Requested-With': 'XMLHttpRequest',
        'Referer': 'https://www.taiwanlottery.com/lotto/history/history_result/'
    }
    now = datetime.now(TW_TZ)
    roc_year = now.year - 1911

    # 台彩新版 AJAX endpoint（嘗試常見格式）
    candidates = [
        f"https://www.taiwanlottery.com/api/Lotto649/GetLotto649HistoryResult?year={roc_year}&month=0",
        f"https://www.taiwanlottery.com/lotto/Lotto649/GetHistoryResult?year={roc_year}",
    ]
    for url in candidates:
        try:
            res = requests.get(url, headers=headers, timeout=20)
            if res.status_code == 200:
                data = res.json()
                # 嘗試解析常見結構
                rows = data if isinstance(data, list) else data.get('data', data.get('result', []))
                for item in rows:
                    try:
                        nums = [int(item.get(f'n{i}', item.get(f'num{i}', 0))) for i in range(1, 8)]
                        if all(1 <= n <= 49 for n in nums) and len(nums) == 7:
                            all_results.append(nums)
                    except Exception:
                        continue
                if all_results:
                    print(f"✅ 備援 API 成功：{len(all_results)} 筆")
                    break
        except Exception as e:
            print(f"備援 {url} 失敗: {e}")

    return all_results


# ─────────────────────────────────────────
# 3. 主流程
# ─────────────────────────────────────────
def save_error(msg, count=0):
    tw_now = datetime.now(TW_TZ).strftime('%Y-%m-%d %H:%M:%S')
    result = {
        "update_time": tw_now,
        "predict_numbers": [],
        "special_number": None,
        "status": f"Error: {msg}",
        "data_count": count
    }
    with open("result.json", "w", encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"⚠️  {msg}")


print("🔍 開始抓取台灣彩券大樂透資料...")
raw_data = fetch_official_csv()

if len(raw_data) < 10:
    print(f"⚠️  CSV 資料不足（{len(raw_data)} 筆），嘗試備援...")
    raw_data += fetch_history_page_backup()

if len(raw_data) < 10:
    save_error(f"資料抓取失敗，僅得 {len(raw_data)} 筆", len(raw_data))
    exit(0)

print(f"📊 共 {len(raw_data)} 筆資料，開始 AI 訓練...")

# ─────────────────────────────────────────
# 4. AI 模型：遺漏值 + RandomForest
# ─────────────────────────────────────────
df = pd.DataFrame(raw_data).iloc[::-1].reset_index(drop=True)  # 由舊到新

last_seen = {i: 0 for i in range(1, 50)}
X_train, y_train = [], []

for _, row in df.iterrows():
    X_train.append(list(last_seen.values()))
    current_set = set(int(v) for v in row if pd.notna(v))
    for i in range(1, 50):
        last_seen[i] = 0 if i in current_set else last_seen[i] + 1
    label = np.zeros(49)
    for n in current_set:
        if 1 <= n <= 49:
            label[n - 1] = 1
    y_train.append(label)

if len(X_train) < 2:
    save_error("訓練資料不足")
    exit(0)

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(np.array(X_train[:-1]), np.array(y_train[1:]))

# 預測
latest_x = np.array([list(last_seen.values())])
probs = np.zeros(49)
for clf in model.estimators_:
    p = clf.predict_proba(latest_x)
    for i in range(49):
        probs[i] += p[i][0][1] if len(p[i][0]) > 1 else 0
probs /= len(model.estimators_)

top_idx = np.argsort(probs)[-7:][::-1]
res_nums = sorted(int(i + 1) for i in top_idx)

# ─────────────────────────────────────────
# 5. 寫出 result.json
# ─────────────────────────────────────────
tw_now = datetime.now(TW_TZ).strftime('%Y-%m-%d %H:%M:%S')
result = {
    "update_time": tw_now,
    "predict_numbers": res_nums[:6],
    "special_number": res_nums[-1],
    "status": "Success",
    "data_count": len(raw_data)
}
with open("result.json", "w", encoding='utf-8') as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

print(f"✅ 預測完成: {result}")
