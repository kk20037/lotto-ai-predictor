import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import json
import time
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from datetime import datetime
import pytz
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
TW_TZ = pytz.timezone('Asia/Taipei')

# ══════════════════════════════════════════════════════════════
# 爬蟲：大樂透 Lotto 6/49 (6+1顆球)
# ══════════════════════════════════════════════════════════════
def fetch_lotto649(pages=50):
    all_results = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    }
    for page in range(1, pages + 1):
        url = f"https://en.lottolyzer.com/history/taiwan/lotto-649/page/{page}"
        print(f"[大樂透] 第 {page}/{pages} 頁（累計 {len(all_results)} 筆）")
        try:
            res = requests.get(url, headers=headers, timeout=20)
            if res.status_code != 200: continue
            soup = BeautifulSoup(res.text, 'html.parser')
            rows = soup.select('table tbody tr')
            for row in rows:
                balls = row.select('td.ball, span.ball, li.ball, td.number, .lottery-ball')
                # 大樂透是 6+1 顆球 = 7 顆
                if len(balls) >= 7:
                    try:
                        nums = [int(b.get_text(strip=True)) for b in balls[:7]]
                        if all(1 <= n <= 49 for n in nums):
                            all_results.append(nums)
                    except: continue
            time.sleep(0.5)
        except Exception as e:
            print(f"  錯誤: {e}")
    print(f"[大樂透] 共 {len(all_results)} 筆")
    return all_results

# ══════════════════════════════════════════════════════════════
# 爬蟲：威力彩 Super Lotto (6+1顆球)
# ══════════════════════════════════════════════════════════════
def fetch_superlotto(pages=50):
    all_results = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    }
    for page in range(1, pages + 1):
        url = f"https://en.lottolyzer.com/history/taiwan/super-lotto/page/{page}"
        print(f"[威力彩] 第 {page}/{pages} 頁（累計 {len(all_results)} 筆）")
        try:
            res = requests.get(url, headers=headers, timeout=20)
            if res.status_code != 200: continue
            soup = BeautifulSoup(res.text, 'html.parser')
            rows = soup.select('table tbody tr')
            for row in rows:
                balls = row.select('td.ball, span.ball, li.ball, td.number, .lottery-ball')
                # ✅ 修正點：威力彩是 6個主號 + 1個特別號 = 7 顆球 (原本寫 8 導致抓不到)
                if len(balls) >= 7:
                    try:
                        nums = [int(b.get_text(strip=True)) for b in balls[:7]]
                        # 驗證：前6顆 1~38，第7顆 1~8
                        if all(1 <= n <= 38 for n in nums[:6]) and (1 <= nums[6] <= 8):
                            all_results.append(nums)
                    except: continue
            time.sleep(0.5)
        except Exception as e:
            print(f"  錯誤: {e}")
    print(f"[威力彩] 共 {len(all_results)} 筆")
    return all_results

# ══════════════════════════════════════════════════════════════
# 特徵工程
# ══════════════════════════════════════════════════════════════
def build_features(raw_data, max_num):
    df = pd.DataFrame(raw_data).iloc[::-1].reset_index(drop=True)
    last_seen  = {i: 0 for i in range(1, max_num + 1)}
    total_freq = {i: 0 for i in range(1, max_num + 1)}
    recent_buf = []
    X_train, y_train = [], []

    for _, row in df.iterrows():
        curr_set = set(row)
        recent_freq = {i: 0 for i in range(1, max_num + 1)}
        for past in recent_buf[-20:]:
            for n in past: recent_freq[n] += 1

        num_list = sorted(list(curr_set))
        num_sum = sum(num_list)
        odd_count = sum(1 for n in num_list if n % 2 == 1)
        small_count = sum(1 for n in num_list if n <= max_num // 2)
        span = max(num_list) - min(num_list) if num_list else 0

        feature = (
            list(last_seen.values()) + 
            list(total_freq.values()) + 
            list(recent_freq.values()) + 
            [num_sum, odd_count, small_count, span]
        )
        X_train.append(feature)

        label = np.zeros(max_num)
        for n in curr_set:
            if 1 <= n <= max_num: label[n-1] = 1
        y_train.append(label)

        for i in range(1, max_num + 1):
            last_seen[i] = 0 if i in curr_set else last_seen[i] + 1
            total_freq[i] += 1 if i in curr_set else 0
        recent_buf.append(curr_set)

    return X_train, y_train, last_seen, total_freq, recent_buf

# ══════════════════════════════════════════════════════════════
# AI 預測
# ══════════════════════════════════════════════════════════════
def predict(raw_data, max_num, pick):
    if len(raw_data) < 10: return [], None
    X_train, y_train, last_seen, total_freq, recent_buf = build_features(raw_data, max_num)
    
    X = np.array(X_train[:-1])
    y = np.array(y_train[1:])

    model = ExtraTreesClassifier(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X, y)

    recent_freq = {i: 0 for i in range(1, max_num + 1)}
    for past in recent_buf[-20:]:
        for n in past: recent_freq[n] += 1

    latest_x = np.array([[*list(last_seen.values()), *list(total_freq.values()), *list(recent_freq.values()), 0, 0, 0, 0]])
    
    probas = model.predict_proba(latest_x)
    probs = np.array([p[0][1] if p[0].shape[0] > 1 else 0 for p in probas])

    top_idx = np.argsort(probs)[-pick:][::-1]
    res_nums = sorted([int(i + 1) for i in top_idx])
    
    # 特別號預測（簡單取近期遺漏最高的）
    special_idx = np.argmax(list(last_seen.values())[:8 if max_num==38 else 49])
    return res_nums, special_idx + 1

# ══════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════
def main():
    now_str = datetime.now(TW_TZ).strftime('%Y-%m-%d %H:%M:%S')
    result = {"update_time": now_str}

    # ── 大樂透 ──
    print("\n🎱 執行大樂透分析...")
    lotto_data = fetch_lotto649(pages=40)
    if len(lotto_data) >= 10:
        nums, special = predict(lotto_data, max_num=49, pick=6)
        result["lotto649"] = {"predict_numbers": nums, "special_number": special, "status": "Success", "data_count": len(lotto_data)}
    else:
        result["lotto649"] = {"predict_numbers": [], "special_number": None, "status": "Error", "data_count": len(lotto_data)}

    # ── 威力彩 ──
    print("\n⚡ 執行威力彩分析...")
    super_data = fetch_superlotto(pages=40)
    if len(super_data) >= 10:
        # ✅ 修正點：威力彩主號是 6 個 (原本寫 pick=7 也是錯的)
        nums, special = predict(super_data, max_num=38, pick=6)
        result["superlotto"] = {"predict_numbers": nums, "special_number": special, "status": "Success", "data_count": len(super_data)}
    else:
        result["superlotto"] = {"predict_numbers": [], "special_number": None, "status": "Error", "data_count": len(super_data)}

    with open("result.json", "w", encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print("\n✅ 分析完成！結果已寫入 result.json")

if __name__ == "__main__":
    main()
