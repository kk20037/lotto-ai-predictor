import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
import json
import time
from datetime import datetime
import pytz

# --- 1. 強化版爬蟲 ---
def crawl_data():
    base_url = "https://top-lottery.com/history/big/"
    all_results = []
    # 模擬真實瀏覽器，避免被網站封鎖
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    for page in range(1, 11): # 抓 10 頁
        try:
            res = requests.get(f"{base_url}{page}", headers=headers, timeout=15)
            res.encoding = 'utf-8'
            soup = BeautifulSoup(res.text, 'html.parser')
            draw_blocks = soup.find_all('div', class_='lg-line')
            
            if not draw_blocks: continue
            
            for block in draw_blocks:
                nums_list = block.find('ul', class_='lg-numbers-small')
                if nums_list:
                    nums = [int(li.get_text(strip=True)) for li in nums_list.find_all('li')]
                    if len(nums) >= 7:
                        all_results.append(nums)
            time.sleep(0.5)
        except Exception as e:
            print(f"Page {page} error: {e}")
            continue
            
    return all_results

# --- 2. 執行流程 ---
raw_data = crawl_data()

if not raw_data or len(raw_data) < 10:
    print("❌ 抓取資料量不足，停止預測。")
    exit(1) # 主動拋出錯誤讓 Actions 顯示失敗

# 簡單 AI 邏輯 (遺漏值分析)
df = pd.DataFrame(raw_data).iloc[::-1] # 由舊到新
last_seen = {i: 0 for i in range(1, 50)}
X_train, y_train = [], []

for _, row in df.iterrows():
    X_train.append(list(last_seen.values()))
    current_set = set(row)
    for i in range(1, 50):
        last_seen[i] = 0 if i in current_set else last_seen[i] + 1
    
    label = np.zeros(49)
    for n in row:
        if 1 <= n <= 49: label[n-1] = 1
    y_train.append(label)

# 訓練模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(np.array(X_train[:-1]), np.array(y_train[1:]))

# 預測
latest_x = np.array([list(last_seen.values())])
probs = np.array([p[0][1] if len(p[0]) > 1 else 0 for p in model.predict_proba(latest_x)])
top_idx = np.argsort(probs)[-7:][::-1]
res_nums = sorted([int(i+1) for i in top_idx])

# --- 3. 儲存 JSON ---
tw_tz = pytz.timezone('Asia/Taipei')
now = datetime.now(tw_tz).strftime('%Y-%m-%d %H:%M:%S')

result = {
    "update_time": now,
    "predict_numbers": res_nums[:6],
    "special_number": res_nums[-1],
    "status": "Success"
}

with open("result.json", "w", encoding='utf-8') as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

print(f"✅ 預測完成: {result}")
