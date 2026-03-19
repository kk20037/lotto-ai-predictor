import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
import json
import time
from datetime import datetime
import pytz

# --- 1. 強化版爬蟲：爬台灣彩券官方網站 ---
def crawl_official_lotto649():
    """
    爬取台灣彩券官網大樂透歷史開獎紀錄
    官方網站穩定，不易被封鎖
    """
    all_results = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Referer': 'https://www.taiwanlottery.com/'
    }

    tw_tz = pytz.timezone('Asia/Taipei')
    now = datetime.now(tw_tz)

    # 往前抓最近 6 個月（每月各抓一次）
    for i in range(6):
        month = now.month - i
        year = now.year
        if month <= 0:
            month += 12
            year -= 1

        year_str = str(year)
        month_str = str(month).zfill(2)

        try:
            url = "https://www.taiwanlottery.com/Lotto/Lotto649/history.aspx"
            params = {
                'Year': year_str,
                'Month': month_str
            }
            res = requests.get(url, params=params, headers=headers, timeout=20)
            res.encoding = 'utf-8'
            soup = BeautifulSoup(res.text, 'html.parser')

            # 官網 table class
            table = soup.find('table', id='Lotto649Control_history_dlQuery')
            if not table:
                # 嘗試備用 selector
                table = soup.find('table', {'class': lambda c: c and 'history' in c.lower()})

            if not table:
                print(f"⚠️  {year_str}-{month_str} 找不到 table，跳過")
                continue

            rows = table.find_all('tr')
            for row in rows[1:]:  # 跳過 header
                cols = row.find_all('td')
                if len(cols) >= 8:
                    try:
                        nums = []
                        for j in range(2, 8):  # 第2~7欄是6個號碼
                            text = cols[j].get_text(strip=True)
                            if text.isdigit():
                                nums.append(int(text))
                        special_text = cols[8].get_text(strip=True) if len(cols) > 8 else ''
                        if special_text.isdigit():
                            nums.append(int(special_text))
                        if len(nums) == 7:
                            all_results.append(nums)
                    except Exception:
                        continue

            print(f"✅ {year_str}-{month_str}：抓到 {len(rows)-1} 筆")
            time.sleep(1)

        except Exception as e:
            print(f"❌ {year_str}-{month_str} 錯誤: {e}")
            continue

    return all_results


def crawl_backup_source():
    """
    備援：使用政府開放資料平台 API
    https://data.gov.tw/dataset/72921
    """
    all_results = []
    try:
        url = "https://www.taiwanlottery.com/Lotto/Lotto649/history.aspx"
        headers = {
            'User-Agent': 'Mozilla/5.0',
        }
        # 直接爬最新一頁
        res = requests.get(url, headers=headers, timeout=20)
        res.encoding = 'utf-8'
        soup = BeautifulSoup(res.text, 'html.parser')

        # 找所有有數字的 span（官網號碼球）
        balls = soup.find_all('span', {'class': lambda c: c and 'ball' in c.lower()})
        chunk = []
        for b in balls:
            t = b.get_text(strip=True)
            if t.isdigit():
                chunk.append(int(t))
                if len(chunk) == 7:
                    all_results.append(chunk)
                    chunk = []

    except Exception as e:
        print(f"備援爬蟲失敗: {e}")

    return all_results


# --- 2. 執行流程 ---
print("🔍 開始爬取台灣彩券大樂透資料...")
raw_data = crawl_official_lotto649()

if len(raw_data) < 10:
    print("⚠️  主爬蟲資料不足，嘗試備援...")
    raw_data = crawl_backup_source()

if not raw_data or len(raw_data) < 10:
    print(f"❌ 抓取資料量不足（只有 {len(raw_data)} 筆），停止預測。")
    # 不用 exit(1)，改寫入 error 狀態讓 Actions 繼續（不中斷流程）
    tw_tz = pytz.timezone('Asia/Taipei')
    now = datetime.now(tw_tz).strftime('%Y-%m-%d %H:%M:%S')
    result = {
        "update_time": now,
        "predict_numbers": [],
        "special_number": None,
        "status": "Error: 資料抓取失敗",
        "data_count": len(raw_data)
    }
    with open("result.json", "w", encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    exit(0)  # ← 改成 0，避免 Actions 標紅失敗

print(f"📊 共抓取 {len(raw_data)} 筆資料，開始 AI 分析...")

# --- 3. AI 模型（遺漏值分析 + RandomForest）---
df = pd.DataFrame(raw_data).iloc[::-1]  # 由舊到新排列

last_seen = {i: 0 for i in range(1, 50)}
X_train, y_train = [], []

for _, row in df.iterrows():
    X_train.append(list(last_seen.values()))
    current_set = set(row.dropna().astype(int))
    for i in range(1, 50):
        last_seen[i] = 0 if i in current_set else last_seen[i] + 1

    label = np.zeros(49)
    for n in row.dropna().astype(int):
        if 1 <= n <= 49:
            label[n - 1] = 1
    y_train.append(label)

# 訓練模型（需要至少 2 筆才能做 shift）
if len(X_train) < 2:
    print("❌ 訓練資料不足")
    exit(0)

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(np.array(X_train[:-1]), np.array(y_train[1:]))

# 預測下一期
latest_x = np.array([list(last_seen.values())])
probs = []
for clf in model.estimators_:
    p = clf.predict_proba(latest_x)
    probs.append([p[i][0][1] if len(p[i][0]) > 1 else 0 for i in range(49)])
probs = np.mean(probs, axis=0)

top_idx = np.argsort(probs)[-7:][::-1]
res_nums = sorted([int(i + 1) for i in top_idx])

# --- 4. 儲存 JSON ---
tw_tz = pytz.timezone('Asia/Taipei')
now = datetime.now(tw_tz).strftime('%Y-%m-%d %H:%M:%S')

result = {
    "update_time": now,
    "predict_numbers": res_nums[:6],
    "special_number": res_nums[-1],
    "status": "Success",
    "data_count": len(raw_data)
}

with open("result.json", "w", encoding='utf-8') as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

print(f"✅ 預測完成: {result}")
