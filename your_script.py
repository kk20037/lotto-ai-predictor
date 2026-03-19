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
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    for page in range(1, 11):
        try:
            res = requests.get(f"{base_url}{page}", headers=headers, timeout=15)
            res.encoding = 'utf-8'
            soup = BeautifulSoup(res.text, 'html.parser')
            draw_blocks = soup.find_all('div', class_='lg-line')
            if not draw_blocks:
                continue
            for block in draw_blocks:
                issue_tag = block.find('span', class_='lg-day')
                issue = issue_tag.get_text(strip=True) if issue_tag else ""
                nums_list = block.find('ul', class_='lg-numbers-small')
                if nums_list:
                    nums = [int(li.get_text(strip=True)) for li in nums_list.find_all('li')]
                    if len(nums) >= 7:
                        all_results.append({"issue": issue, "nums": nums})
            time.sleep(0.5)
        except Exception as e:
            print(f"Error crawling page {page}: {e}")
            continue
    return pd.DataFrame(all_results).iloc[::-1].reset_index(drop=True)

# --- 2. 特徵工程與預測 ---
df_raw = crawl_data()
if df_raw.empty:
    raise ValueError("未能抓取到任何資料，請檢查爬蟲邏輯或目標網站。")

last_seen = {i: 0 for i in range(1, 50)}
X_list, y_list = [], []

for idx, row in df_raw.iterrows():
    X_list.append(list(last_seen.values()))
    for i in range(1, 50):
        last_seen[i] = 0 if i in row['nums'] else last_seen[i] + 1
    
    label = np.zeros(49)
    # 大樂透取前 7 碼 (含特別號)
    for n in row['nums'][:7]:
        if 1 <= n <= 49:
            label[n-1] = 1
    y_list.append(label)

X = np.array(X_list[:-1])
y = np.array(y_list[1:])

# 隨機森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 預測下一期 (使用最後一期的遺漏值作為輸入)
latest_x = np.array([list(last_seen.values())])
# 取得各號碼出現機率
probs = np.array([p[0][1] if len(p[0]) > 1 else 0 for p in model.predict_proba(latest_x)])

# --- 3. 取得號碼並輸出 JSON ---
top_idx = np.argsort(probs)[-7:][::-1]
res_nums = sorted([int(i+1) for i in top_idx])

result = {
    "update_time": pd.Timestamp.now(tz='Asia/Taipei').strftime('%Y-%m-%d %H:%M:%S'),
    "predict_numbers": res_nums[:6],
    "special_number": res_nums[-1],
    "status": "AI Prediction Complete"
}

with open("result.json", "w", encoding='utf-8') as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

print("✅ JSON 產出成功:", result)
