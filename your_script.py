import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import json
import time
from sklearn.ensemble import ExtraTreesClassifier
from datetime import datetime
import pytz
import urllib3

# 關閉 SSL 警告 (GitHub Actions 常見需求)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
TW_TZ = pytz.timezone('Asia/Taipei')

# ══════════════════════════════════════════════════════════════
# 修正版爬蟲：對接官方 API (最穩定)
# ══════════════════════════════════════════════════════════════
def fetch_lotto649(pages=1):
    all_results = []
    url = "https://www.taiwanlottery.com.tw/app/data/last_60_Lotto649.json"
    print(f"[大樂透] 正在從台彩官網 API 抓取資料...")
    try:
        res = requests.get(url, verify=False, timeout=20)
        res.encoding = 'utf-8'
        data = res.json()
        for item in data.get('data', []):
            nums = [int(n) for n in item.get('no', [])] # 轉為整數
            s_num = item.get('sno')
            if len(nums) == 6 and s_num:
                all_results.append(nums + [int(s_num)])
    except Exception as e:
        print(f"  大樂透 API 錯誤: {e}")
    return all_results

def fetch_superlotto(pages=1):
    all_results = []
    url = "https://www.taiwanlottery.com.tw/app/data/last_60_SuperLotto.json"
    print(f"[威力彩] 正在從台彩官網 API 抓取資料...")
    try:
        res = requests.get(url, verify=False, timeout=20)
        res.encoding = 'utf-8'
        data = res.json()
        for item in data.get('data', []):
            nums = [int(n) for n in item.get('no', [])]
            s_num = item.get('sno')
            if len(nums) == 6 and s_num:
                all_results.append(nums + [int(s_num)])
    except Exception as e:
        print(f"  威力彩 API 錯誤: {e}")
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

    # 使用穩定性更高的 ExtraTrees
    model = ExtraTreesClassifier(n_estimators=1000, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X, y)

    recent_freq = {i: 0 for i in range(1, max_num + 1)}
    for past in recent_buf[-20:]:
        for n in past: recent_freq[n] += 1

    latest_x = np.array([[*list(last_seen.values()), *list(total_freq.values()), *list(recent_freq.values()), 0, 0, 0, 0]])
    
    probas = model.predict_proba(latest_x)
    # 修正：確保處理機率維度
    probs = []
    for p in probas:
        if isinstance(p, list): # 多標籤情況處理
            probs.append(p[0][1] if len(p[0]) > 1 else 0)
        else:
            probs.append(p[0][1] if p.shape[1] > 1 else 0)
    
    probs = np.array(probs)
    top_idx = np.argsort(probs)[-pick:][::-1]
    res_nums = sorted([int(i + 1) for i in top_idx])
    
    # 特別號：取目前遺漏次數最高的
    special_limit = 8 if max_num == 38 else 49
    last_seen_list = list(last_seen.values())[:special_limit]
    special_num = np.argmax(last_seen_list) + 1
    
    return res_nums, int(special_num)

# ══════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════
def main():
    now_str = datetime.now(TW_TZ).strftime('%Y-%m-%d %H:%M:%S')
    result = {"update_time": now_str}

    # ── 大樂透 ──
    lotto_data = fetch_lotto649()
    if len(lotto_data) >= 10:
        nums, special = predict(lotto_data, max_num=49, pick=6)
        result["lotto649"] = {"predict_numbers": nums, "special_number": special, "status": "Success", "data_count": len(lotto_data)}
    else:
        result["lotto649"] = {"predict_numbers": [], "special_number": None, "status": "Error", "data_count": len(lotto_data)}

    # ── 威力彩 ──
    super_data = fetch_superlotto()
    if len(super_data) >= 10:
        nums, special = predict(super_data, max_num=38, pick=6)
        result["superlotto"] = {"predict_numbers": nums, "special_number": special, "status": "Success", "data_count": len(super_data)}
    else:
        result["superlotto"] = {"predict_numbers": [], "special_number": None, "status": "Error", "data_count": len(super_data)}

    with open("result.json", "w", encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    print(f"✅ 更新完成！更新時間: {now_str}")
    print(f"📊 大樂透筆數: {len(lotto_data)} | 威力彩筆數: {len(super_data)}")

if __name__ == "__main__":
    main()
