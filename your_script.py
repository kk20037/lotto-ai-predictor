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
# 爬蟲：大樂透 Lotto 6/49
# ══════════════════════════════════════════════════════════════
def fetch_lotto649(pages=50):
    all_results = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    for page in range(1, pages + 1):
        url = f"https://en.lottolyzer.com/history/taiwan/lotto-649/page/{page}"
        print(f"[大樂透] 第 {page}/{pages} 頁（累計 {len(all_results)} 筆）")
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
            time.sleep(0.6)
        except Exception as e:
            print(f"  錯誤: {e}")
    print(f"[大樂透] 共 {len(all_results)} 筆")
    return all_results


# ══════════════════════════════════════════════════════════════
# 爬蟲：威力彩 Super Lotto
# ══════════════════════════════════════════════════════════════
def fetch_superlotto(pages=50):
    all_results = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    for page in range(1, pages + 1):
        url = f"https://en.lottolyzer.com/history/taiwan/super-lotto/page/{page}"
        print(f"[威力彩] 第 {page}/{pages} 頁（累計 {len(all_results)} 筆）")
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
            time.sleep(0.6)
        except Exception as e:
            print(f"  錯誤: {e}")
    print(f"[威力彩] 共 {len(all_results)} 筆")
    return all_results


# ══════════════════════════════════════════════════════════════
# 特徵工程：遺漏值 + 頻率 + 近期熱度 + 統計特徵
# ══════════════════════════════════════════════════════════════
def build_features(raw_data, max_num):
    df = pd.DataFrame(raw_data).iloc[::-1].reset_index(drop=True)

    last_seen  = {i: 0 for i in range(1, max_num + 1)}  # 遺漏值
    total_freq = {i: 0 for i in range(1, max_num + 1)}  # 歷史總頻率
    recent_buf = []                                       # 近期開獎記錄

    X_train, y_train = [], []

    for _, row in df.iterrows():
        curr_set = set()
        for v in row:
            try:
                n = int(v)
                if 1 <= n <= max_num:
                    curr_set.add(n)
            except Exception:
                pass

        if not curr_set:
            continue

        # 近 20 期熱度
        recent_freq = {i: 0 for i in range(1, max_num + 1)}
        for past in recent_buf[-20:]:
            for n in past:
                recent_freq[n] += 1

        # 統計特徵
        num_list   = sorted(curr_set)
        num_sum    = sum(num_list)
        odd_count  = sum(1 for n in num_list if n % 2 == 1)
        small_count = sum(1 for n in num_list if n <= max_num // 2)
        span       = max(num_list) - min(num_list) if num_list else 0

        feature = (
            list(last_seen.values())   +   # 遺漏值      max_num 維
            list(total_freq.values())  +   # 歷史頻率    max_num 維
            list(recent_freq.values()) +   # 近期熱度    max_num 維
            [num_sum, odd_count, small_count, span]  # 統計特徵 4 維
        )
        X_train.append(feature)

        label = np.zeros(max_num)
        for n in curr_set:
            label[n - 1] = 1
        y_train.append(label)

        # 更新狀態
        for i in range(1, max_num + 1):
            last_seen[i]  = 0 if i in curr_set else last_seen[i] + 1
            total_freq[i] += 1 if i in curr_set else 0
        recent_buf.append(curr_set)

    return X_train, y_train, last_seen, total_freq, recent_buf


# ══════════════════════════════════════════════════════════════
# AI 預測：RandomForest + ExtraTrees Ensemble
# ══════════════════════════════════════════════════════════════
def predict(raw_data, max_num, pick):
    if len(raw_data) < 10:
        return [], None

    X_train, y_train, last_seen, total_freq, recent_buf = build_features(raw_data, max_num)

    if len(X_train) < 2:
        return [], None

    X = np.array(X_train[:-1])
    y = np.array(y_train[1:])

    # 兩個模型
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    et = ExtraTreesClassifier(
        n_estimators=300, max_depth=10,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )

    print("  🤖 訓練 RandomForest...")
    rf.fit(X, y)
    print("  🤖 訓練 ExtraTrees...")
    et.fit(X, y)

    # 最新一期特徵
    recent_freq = {i: 0 for i in range(1, max_num + 1)}
    for past in recent_buf[-20:]:
        for n in past:
            recent_freq[n] += 1

    latest_x = np.array([[
        *list(last_seen.values()),
        *list(total_freq.values()),
        *list(recent_freq.values()),
        0, 0, 0, 0   # 預測時統計特徵補 0
    ]])

    def get_probs(model):
        probas = model.predict_proba(latest_x)
        return np.array([p[0][1] if p[0].shape[0] > 1 else 0 for p in probas])

    # 兩個模型機率平均（Ensemble）
    probs = (get_probs(rf) + get_probs(et)) / 2.0

    top_idx  = np.argsort(probs)[-(pick + 1):][::-1]
    res_nums = sorted([int(i + 1) for i in top_idx])
    return res_nums[:pick], res_nums[-1]


# ══════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════
def main():
    now_str = datetime.now(TW_TZ).strftime('%Y-%m-%d %H:%M:%S')
    result  = {"update_time": now_str}

    # ── 大樂透 ──────────────────────────────────────────────
    print("\n🎱 開始抓取大樂透資料...")
    lotto_data = fetch_lotto649(pages=50)

    if len(lotto_data) >= 10:
        print(f"\n📊 大樂透 {len(lotto_data)} 筆，開始訓練...")
        nums, special = predict(lotto_data, max_num=49, pick=6)
        result["lotto649"] = {
            "predict_numbers": nums,
            "special_number":  special,
            "status":          "Success",
            "data_count":      len(lotto_data)
        }
        print(f"✅ 大樂透預測：{nums}  特別號：{special}")
    else:
        result["lotto649"] = {
            "predict_numbers": [],
            "special_number":  None,
            "status":          f"Error: 僅抓到 {len(lotto_data)} 筆",
            "data_count":      len(lotto_data)
        }
        print(f"❌ 大樂透資料不足")

    # ── 威力彩 ──────────────────────────────────────────────
    print("\n⚡ 開始抓取威力彩資料...")
    super_data = fetch_superlotto(pages=50)

    if len(super_data) >= 10:
        print(f"\n📊 威力彩 {len(super_data)} 筆，開始訓練...")
        nums, special = predict(super_data, max_num=38, pick=7)
        result["superlotto"] = {
            "predict_numbers": nums,
            "special_number":  special,
            "status":          "Success",
            "data_count":      len(super_data)
        }
        print(f"✅ 威力彩預測：{nums}  第二區：{special}")
    else:
        result["superlotto"] = {
            "predict_numbers": [],
            "special_number":  None,
            "status":          f"Error: 僅抓到 {len(super_data)} 筆",
            "data_count":      len(super_data)
        }
        print(f"❌ 威力彩資料不足")

    # ── 寫入 JSON ────────────────────────────────────────────
    with open("result.json", "w", encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 完成！")
    print(json.dumps(result, indent=2, ensure_ascii=False))


main()
