import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import json
import time
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import pytz
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
TW_TZ = pytz.timezone('Asia/Taipei')

def fetch_lottolyzer(pages=15):
    all_results = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    for page in range(1, pages + 1):
        url = f"https://en.lottolyzer.com/history/taiwan/lotto-649/page/{page}"
        print(f"📡 第 {page} 頁: {url}")
        try:
            res = requests.get(url, headers=headers, timeout=20)
            if res.status_code != 200:
                print(f"   ❌ HTTP {res.status_code}")
                continue

            soup = BeautifulSoup(res.text, 'html.parser')
            rows = soup.select('table tbody tr')
            if not rows:
                rows = soup.select('tr.result-row, tr[data-draw]')

            count = 0
            for row in rows:
                balls = row.select('td.ball, span.ball, li.ball, td.number, .lottery-ball')
                if len(balls) >= 7:
                    try:
                        nums = [int(b.get_text(strip=True)) for b in balls[:7]]
                        if all(1 <= n <= 49 for n in nums):
                            all_results.append(nums)
                            count += 1
                    except Exception:
                        continue
                elif len(balls) == 0:
                    tds = row.find_all('td')
                    nums = []
                    for td in tds:
                        t = td.get_text(strip=True)
                        if t.isdigit() and 1 <= int(t) <= 49:
                            nums.append(int(t))
                    if len(nums) == 7:
                        all_results.append(nums)
                        count += 1

            print(f"   ✅ 本頁取得 {count} 筆（累計 {len(all_results)} 筆）")
            time.sleep(0.8)

        except Exception as e:
            print(f"   💥 錯誤: {e}")
            continue

    return all_results


def main():
    print("🔍 開始抓取大樂透歷史資料...\n")
    raw_data = fetch_lottolyzer(pages=15)
    now_str = datetime.now(TW_TZ).strftime('%Y-%m-%d %H:%M:%S')

    if len(raw_data) < 10:
        print(f"\n❌ 資料量不足（{len(raw_data)} 筆）")
        result = {
            "update_time": now_str,
            "predict_numbers": [],
            "special_number": None,
            "status": f"Error: 僅抓到 {len(raw_data)} 筆",
            "data_count": len(raw_data)
        }
        with open("result.json", "w", encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        return

    print(f"\n📊 共 {len(raw_data)} 筆資料，開始 AI 分析...")

    df = pd.DataFrame(raw_data).iloc[::-1].reset_index(drop=True)

    last_seen = {i: 0 for i in range(1, 50)}
    X_train, y_train = [], []

    for _, row in df.iterrows():
        X_train.append(list(last_seen.values()))
        curr_set = set()
        for v in row:
            try:
                n = int(v)
                if 1 <= n <= 49:
                    curr_set.add(n)
            except Exception:
                pass
        for i in range(1, 50):
            last_seen[i] = 0 if i in curr_set else last_seen[i] + 1
        label = np.zeros(49)
        for n in curr_set:
            label[n - 1] = 1
        y_train.append(label)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(np.array(X_train[:-1]), np.array(y_train[1:]))

    latest_x = np.array([list(last_seen.values())])
    all_probs = model.predict_proba(latest_x)
    probs = np.array([p[0][1] if p[0].shape[0] > 1 else 0 for p in all_probs])

    top_idx = np.argsort(probs)[-7:][::-1]
    res_nums = sorted([int(i + 1) for i in top_idx])

    result = {
        "update_time": now_str,
        "predict_numbers": res_nums[:6],
        "special_number": res_nums[-1],
        "status": "Success",
        "data_count": len(raw_data)
    }

    with open("result.json", "w", encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 預測完成: {result}")


main()
