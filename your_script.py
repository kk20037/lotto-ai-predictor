import pandas as pd
import numpy as np
import requests
import json
import time
from sklearn.ensemble import ExtraTreesClassifier
from datetime import datetime
import pytz
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
TW_TZ = pytz.timezone('Asia/Taipei')

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# ══════════════════════════════════════════════════════════════
# 爬蟲：對接 api.taiwanlottery.com（已驗證可用）
# ══════════════════════════════════════════════════════════════

def fetch_lotto649(months=8):
    """
    大樂透 6/49
    API: Lotto649Result
    回傳 key: content -> lotto649Res -> drawNumberSize (7個, 前6主號+第7特別號)
    """
    all_results = []
    now = datetime.now(TW_TZ)
    url = "https://api.taiwanlottery.com/TLCAPIWeB/Lottery/Lotto649Result"

    for i in range(months):
        month_str = (now - pd.DateOffset(months=i)).strftime('%Y-%m')
        params = {"month": month_str, "pageNum": 1, "pageSize": 50}
        print(f"[大樂透] 抓取 {month_str}...")
        try:
            res = requests.get(url, params=params, headers=HEADERS, verify=False, timeout=20)
            if res.status_code == 200:
                data = res.json()
                results = data.get("content", {}).get("lotto649Res") or []
                for item in results:
                    nums = item.get("drawNumberSize", [])
                    if len(nums) == 7:
                        all_results.append(nums)  # [n1,n2,n3,n4,n5,n6, special]
            time.sleep(0.5)
        except Exception as e:
            print(f"  {month_str} 失敗: {e}")

    print(f"[大樂透] 共 {len(all_results)} 筆")
    return all_results


def fetch_superlotto(months=8):
    """
    威力彩 6/38
    API: SuperLotto638Result  ← 正確路徑（含638）
    回傳 key: content -> superLotto638Res -> drawNumberSize (7個, 前6主號+第7特別號)
    """
    all_results = []
    now = datetime.now(TW_TZ)
    url = "https://api.taiwanlottery.com/TLCAPIWeB/Lottery/SuperLotto638Result"

    for i in range(months):
        month_str = (now - pd.DateOffset(months=i)).strftime('%Y-%m')
        params = {"month": month_str, "pageNum": 1, "pageSize": 50}
        print(f"[威力彩] 抓取 {month_str}...")
        try:
            res = requests.get(url, params=params, headers=HEADERS, verify=False, timeout=20)
            if res.status_code == 200:
                data = res.json()
                results = data.get("content", {}).get("superLotto638Res") or []
                for item in results:
                    nums = item.get("drawNumberSize", [])
                    if len(nums) == 7:
                        all_results.append(nums)
            time.sleep(0.5)
        except Exception as e:
            print(f"  {month_str} 失敗: {e}")

    print(f"[威力彩] 共 {len(all_results)} 筆")
    return all_results


# ══════════════════════════════════════════════════════════════
# 特徵工程
# ══════════════════════════════════════════════════════════════

def build_features(raw_data, max_num):
    # raw_data 每筆 7 個數字，只取前 6 個主號作為訓練特徵
    main_data = [row[:6] for row in raw_data]
    df = pd.DataFrame(main_data).iloc[::-1].reset_index(drop=True)

    last_seen  = {i: 0 for i in range(1, max_num + 1)}
    total_freq = {i: 0 for i in range(1, max_num + 1)}
    recent_buf = []
    X_train, y_train = [], []

    for _, row in df.iterrows():
        curr_set = set(int(v) for v in row if 1 <= int(v) <= max_num)
        recent_freq = {i: 0 for i in range(1, max_num + 1)}
        for past in recent_buf[-20:]:
            for n in past:
                recent_freq[n] += 1

        num_list    = sorted(curr_set)
        num_sum     = sum(num_list)
        odd_count   = sum(1 for n in num_list if n % 2 == 1)
        small_count = sum(1 for n in num_list if n <= max_num // 2)
        span        = max(num_list) - min(num_list) if num_list else 0

        feature = (
            list(last_seen.values()) +
            list(total_freq.values()) +
            list(recent_freq.values()) +
            [num_sum, odd_count, small_count, span]
        )
        X_train.append(feature)

        label = np.zeros(max_num)
        for n in curr_set:
            label[n - 1] = 1
        y_train.append(label)

        for i in range(1, max_num + 1):
            last_seen[i]  = 0 if i in curr_set else last_seen[i] + 1
            total_freq[i] += 1 if i in curr_set else 0
        recent_buf.append(curr_set)

    return X_train, y_train, last_seen, total_freq, recent_buf


# ══════════════════════════════════════════════════════════════
# AI 預測
# ══════════════════════════════════════════════════════════════

def predict(raw_data, max_num, pick):
    if len(raw_data) < 10:
        return [], None

    X_train, y_train, last_seen, total_freq, recent_buf = build_features(raw_data, max_num)

    X = np.array(X_train[:-1])
    y = np.array(y_train[1:])

    model = ExtraTreesClassifier(n_estimators=1000, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X, y)

    recent_freq = {i: 0 for i in range(1, max_num + 1)}
    for past in recent_buf[-20:]:
        for n in past:
            recent_freq[n] += 1

    latest_x = np.array([[
        *list(last_seen.values()),
        *list(total_freq.values()),
        *list(recent_freq.values()),
        0, 0, 0, 0
    ]])

    probas = model.predict_proba(latest_x)
    probs = []
    for p in probas:
        if isinstance(p, list):
            probs.append(p[0][1] if len(p[0]) > 1 else 0)
        else:
            probs.append(p[0][1] if p.shape[1] > 1 else 0)

    probs = np.array(probs)
    top_idx = np.argsort(probs)[-pick:][::-1]
    res_nums = sorted([int(i + 1) for i in top_idx])

    # 特別號：取遺漏次數最高（威力彩特別號範圍 1-8，大樂透 1-49）
    special_limit = 8 if max_num == 38 else 49
    last_seen_list = list(last_seen.values())[:special_limit]
    special_num = int(np.argmax(last_seen_list) + 1)

    return res_nums, special_num


# ══════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════

def main():
    now_str = datetime.now(TW_TZ).strftime('%Y-%m-%d %H:%M:%S')
    result = {"update_time": now_str}

    # ── 大樂透 ──
    lotto_data = fetch_lotto649(months=8)
    if len(lotto_data) >= 10:
        nums, special = predict(lotto_data, max_num=49, pick=6)
        result["lotto649"] = {
            "predict_numbers": nums,
            "special_number":  special,
            "status":          "Success",
            "data_count":      len(lotto_data)
        }
    else:
        result["lotto649"] = {
            "predict_numbers": [],
            "special_number":  None,
            "status":          "Error",
            "data_count":      len(lotto_data)
        }

    # ── 威力彩 ──
    super_data = fetch_superlotto(months=8)
    if len(super_data) >= 10:
        nums, special = predict(super_data, max_num=38, pick=6)
        result["superlotto"] = {
            "predict_numbers": nums,
            "special_number":  special,
            "status":          "Success",
            "data_count":      len(super_data)
        }
    else:
        result["superlotto"] = {
            "predict_numbers": [],
            "special_number":  None,
            "status":          "Error",
            "data_count":      len(super_data)
        }

    with open("result.json", "w", encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"\n更新完成！時間更新: {now_str}")
    print(f"📊 大樂透筆數: {len(lotto_data)} | 威力彩筆數: {len(super_data)}")
    print(f"🎯 大樂透: {result['lotto649']['predict_numbers']} 特別號: {result['lotto649']['special_number']}")
    print(f"🎯 威力彩: {result['superlotto']['predict_numbers']} 特別號: {result['superlotto']['special_number']}")


if __name__ == "__main__":
    main()
