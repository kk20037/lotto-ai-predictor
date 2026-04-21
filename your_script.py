import pandas as pd
import numpy as np
import requests
import json
import time
from sklearn.ensemble import ExtraTreesClassifier
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytz
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
TW_TZ = pytz.timezone('Asia/Taipei')
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# ══════════════════════════════════════════════════════════════
# 工具：產生過去 N 個月的 YYYY-MM 字串清單
# ══════════════════════════════════════════════════════════════
def get_month_list(months=8):
    now = datetime.now(TW_TZ)
    return [(now - relativedelta(months=i)).strftime('%Y-%m') for i in range(months)]

# ══════════════════════════════════════════════════════════════
# 爬蟲
# 大樂透：Lotto649Result        → lotto649Res
# 威力彩：SuperLotto638Result   → superLotto638Res
# 每筆 drawNumberSize = [n1,n2,n3,n4,n5,n6, special]  共 7 個
# ══════════════════════════════════════════════════════════════
def fetch_data(url, content_key, months=8):
    all_results = []
    for month_str in get_month_list(months):
        params = {"month": month_str, "pageNum": 1, "pageSize": 50}
        try:
            res = requests.get(url, params=params, headers=HEADERS, verify=False, timeout=20)
            if res.status_code == 200:
                items = res.json().get("content", {}).get(content_key) or []
                for item in items:
                    nums = item.get("drawNumberSize", [])
                    if len(nums) == 7:
                        all_results.append([int(n) for n in nums])
            time.sleep(0.3)
        except Exception as e:
            print(f"  {month_str} 失敗: {e}")
    return all_results

def fetch_lotto649(months=8):
    print("[大樂透] 抓取資料...")
    data = fetch_data(
        "https://api.taiwanlottery.com/TLCAPIWeB/Lottery/Lotto649Result",
        "lotto649Res",
        months
    )
    print(f"[大樂透] 共 {len(data)} 筆")
    return data

def fetch_superlotto(months=8):
    print("[威力彩] 抓取資料...")
    data = fetch_data(
        "https://api.taiwanlottery.com/TLCAPIWeB/Lottery/SuperLotto638Result",
        "superLotto638Res",
        months
    )
    print(f"[威力彩] 共 {len(data)} 筆")
    return data

# ══════════════════════════════════════════════════════════════
# 特徵工程（只用前 6 個主號，特別號另外處理）
# ══════════════════════════════════════════════════════════════
def build_features(raw_data, max_num):
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
        span        = (max(num_list) - min(num_list)) if num_list else 0

        X_train.append(
            list(last_seen.values()) +
            list(total_freq.values()) +
            list(recent_freq.values()) +
            [num_sum, odd_count, small_count, span]
        )

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
        probs.append(p[0][1] if p.shape[1] > 1 else 0)

    probs = np.array(probs)
    top_idx = np.argsort(probs)[-pick:][::-1]
    res_nums = sorted([int(i + 1) for i in top_idx])

    # 特別號：從原始資料取特別號遺漏統計
    # 威力彩特別號範圍 1-8，大樂透 1-49
    special_limit = 8 if max_num == 38 else 49
    special_seen = {i: 0 for i in range(1, special_limit + 1)}
    for row in reversed(raw_data):
        s = int(row[6])
        if 1 <= s <= special_limit:
            special_seen[s] = 0  # 出現就重置
        for k in special_seen:
            if k != s:
                special_seen[k] += 1
        break  # 只更新最後一期

    # 用遺漏最久的特別號（從全部資料統計）
    special_last = {i: 0 for i in range(1, special_limit + 1)}
    for row in reversed(raw_data):
        s = int(row[6])
        for k in range(1, special_limit + 1):
            if k == s:
                if special_last[k] == 0:
                    special_last[k] = 0  # 已出現
            else:
                special_last[k] += 1

    # 簡單統計：哪個特別號最久沒出現
    special_counter = {i: 0 for i in range(1, special_limit + 1)}
    for row in raw_data:
        s = int(row[6])
        if 1 <= s <= special_limit:
            special_counter[s] += 1

    # 取出現頻率最低的（遺漏最多）
    special_num = int(min(special_counter, key=special_counter.get))

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

    print(f"\n✅ 更新完成！更新時間: {now_str}")
    print(f"📊 大樂透筆數: {len(lotto_data)} | 威力彩筆數: {len(super_data)}")
    print(f"🎯 大樂透: {result['lotto649']['predict_numbers']} 特別號: {result['lotto649']['special_number']}")
    print(f"🎯 威力彩: {result['superlotto']['predict_numbers']} 特別號: {result['superlotto']['special_number']}")

if __name__ == "__main__":
    main()
