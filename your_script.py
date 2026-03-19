import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
import json
import time

# --- 1. 抓取資料 (抓最近 10 頁確保雲端執行效率) ---
def crawl_data():
    base_url = "https://top-lottery.com/history/big/"
    all_results = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    for page in range(1, 11):
        try:
            res = requests.get(f"{base_url}{page}", headers=headers)
            res.encoding = 'utf-8'
            soup = BeautifulSoup(res.text, 'html.parser')
            for block in soup.find_all('div', class_='lg-line'):
                issue = block.find('span', class_='lg-day').get_text(strip=True)
                nums_list = block.find('ul', class_='lg-numbers-small')
                if nums_list:
                    nums = [int(li.get_text(strip=True)) for li in nums_list.find_all('li')]
                    all_results.append({"issue": issue, "nums": nums})
            time.sleep(0.5)
        except: continue
    return pd.DataFrame(all_results).iloc[::-1].reset_index(drop=True)

# --- 2. 特徵工程與預測 ---
df = crawl_data()
last_seen = {i: 0 for i in range(1, 50)}
X_list, y_list = [], []

for idx, row in df.iterrows():
    X_list.append(list(last_seen.values()))
    for i in range(1, 50):
        last_seen[i] = 0 if i in row['nums'] else last_seen[i] + 1
    label = np.zeros(49)
    label[[n-1 for n in row['nums']]] = 1
    y_list.append(label)

X = np.array(X_list[:-1])
y = np.array(y_list[1:])
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 預測下一期
latest_x = np.array([list(last_seen.values())])
probs = np.array([p[0][1] if len(p[0]) > 1 else 0 for p in model.predict_proba(latest_x)])

# --- 3. 取得號碼並輸出 JSON ---
top_idx = np.argsort(probs)[-7:][::-1]
res_nums = sorted([int(i+1) for i in top_idx])

result = {
    "update_time": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    "predict_numbers": res_nums[:6],
    "special_number": res_nums[-1],
    "status": "AI Prediction Complete"
}

with open("result.json", "w", encoding='utf-8') as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

print("✅ JSON 產出成功:", result)
