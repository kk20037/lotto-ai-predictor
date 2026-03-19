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

# ══════════════════════════════════════════════════════════════
# 大樂透 Lotto 6/49
# ══════════════════════════════════════════════════════════════
def fetch_lotto649(pages=15):
    all_results = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    for page in range(1, pages + 1):
        url = f"https://en.lottolyzer.com/history/taiwan/lotto-649/page/{page}"
        print(f"[大樂透] 第 {page} 頁")
        try:
            res = requests.get(url, headers=headers, timeout=20)
            if res.status_code != 200:
                continue
            soup = BeautifulSoup(res.text, 'html.parser')
            rows = soup.select('table tbody tr')
            for row in rows:
                balls = row.select('td.ball, span.ball, li.ball, td.number, .lottery-ball')
                if len(balls) >= 7:
                    try:
                        nums = [int(b.get_text(strip=True)) for b in balls[:7]]
                        if all(1 <= n <= 49 for n in nums):
                            all_results.append(nums)
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
            time.sleep(0.8)
        except Exception as e:
            print(f"  錯誤: {e}")
    print(f"[大樂透] 共 {len(all_results)} 筆")
    return all_results


# ══════════════════════════════════════════════════════════════
# 威力彩 Super Lotto (8/38 + 特別號 1/8)
# 前 8 球範圍 1~38，特別號 1~8
# ══════════════════════════════════════════════════════════════
def fetch_superlotto(pages=15):
    all_results = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    for page in range(1, pages + 1):
        url = f"https://en.lottolyzer.com/history/taiwan/super-lotto/page/{page}"
        print(f"[威力彩] 第 {page} 頁")
        try:
            res = requests.get(url, headers=headers, timeout=20)
            if res.status_code != 200:
                continue
            soup = BeautifulSoup(res.text, 'html.parser')
            rows = soup.select('table tbody tr')
            for row in rows:
                balls = row.select('td.ball, span.ball, li.ball, td.number, .lottery-ball')
                if len(balls) >= 8:
                    try:
                        nums = [int(b.get_text(strip=True)) for b in balls[:8]]
                        # 前7球 1~38，第8球(特別號) 1~8
                        if all(1 <= n <= 38 for n in nums[:7]) and 1 <= nums[7] <= 8:
                            all_results.append(nums)
                    except Exception:
                        continue
                elif len(balls) == 0:
                    tds = row.find_all('td')
                    nums = []
                    for td in tds:
                        t = td.get_text(strip=True)
                        if t.isdigit():
                            nums.append(int(t))
                    if len(nums) == 8:
                        all_results.append(nums)
            time.sleep(0.8)
        except Exception as e:
            print(f"  錯誤: {e}")
    print(f"[威力彩] 共 {len(all_results)} 筆")
    return all_results


# ══════════════════════════════════════════════════════════════
# AI 預測（通用）
# max_num: 號碼最大值, pick: 取幾顆主球
# ══════════════════════════════════════════════════════════════
def predict(raw_data, max_num, pick):
    df = pd.DataFrame(raw_data).iloc[::-1].reset_index(drop=True)
    last_seen = {i: 0 for i in range(1, max_num + 1)}
    X_train, y_train = [], []

    for _, row in df.iterrows():
        X_train.append(list(last_seen.values()))
        curr_set = set()
        for v in row:
            try:
                n = int(v)
                if 1 <= n <= max_num:
                    curr_set.add(n)
            except Exception:
                pass
        for i in range(1, max_num + 1):
            last_seen[i] = 0 if i in curr_set else last_seen[i] + 1
        label = np.zeros(max_num)
        for n in curr_set:
            label[n - 1] = 1
        y_train.append(label)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(np.array(X_train[:-1]), np.array(y_train[1:]))

    latest_x = np.array([list(last_seen.values())])
    all_probs = model.predict_proba(latest_x)
    probs = np.array([p[0][1] if p[0].shape[0] > 1 else 0 for p in all_probs])

    top_idx = np.argsort(probs)[-(pick + 1):][::-1]
    res_nums = sorted([int(i + 1) for i in top_idx])
    return res_nums[:pick], res_nums[-1]


# ══════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════
def main():
    now_str = datetime.now(TW_TZ).strftime('%Y-%m-%d %H:%M:%S')
    result = {"update_time": now_str}

    # ── 大樂透 ──
    print("\n🎱 開始抓取大樂透資料...")
    lotto_data = fetch_lotto649(pages=15)
    if len(lotto_data) >= 10:
        nums, special = predict(lotto_data, max_num=49, pick=6)
        result["lotto649"] = {
            "predict_numbers": nums,
            "special_number": special,
            "status": "Success",
            "data_count": len(lotto_data)
        }
        print(f"✅ 大樂透預測：{nums} 特別號：{special}")
    else:
        result["lotto649"] = {
            "predict_numbers": [],
            "special_number": None,
            "status": f"Error: 僅抓到 {len(lotto_data)} 筆",
            "data_count": len(lotto_data)
        }

    # ── 威力彩 ──
    print("\n⚡ 開始抓取威力彩資料...")
    super_data = fetch_superlotto(pages=15)
    if len(super_data) >= 10:
        nums, special = predict(super_data, max_num=38, pick=7)
        result["superlotto"] = {
            "predict_numbers": nums,
            "special_number": special,
            "status": "Success",
            "data_count": len(super_data)
        }
        print(f"✅ 威力彩預測：{nums} 第二區：{special}")
    else:
        result["superlotto"] = {
            "predict_numbers": [],
            "special_number": None,
            "status": f"Error: 僅抓到 {len(super_data)} 筆",
            "data_count": len(super_data)
        }

    with open("result.json", "w", encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 完成！結果已寫入 result.json")
    print(json.dumps(result, indent=2, ensure_ascii=False))


main()
