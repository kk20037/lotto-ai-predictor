import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import json
import time
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.multioutput import MultiOutputClassifier
from datetime import datetime
import pytz
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
TW_TZ = pytz.timezone('Asia/Taipei')

def fetch_lotto649(pages=50):  # ← 從 15 改成 50，抓更多資料
    all_results = []
    headers = {'User-Agent': 'Mozilla/5.0 Chrome/124.0.0.0 Safari/537.36'}
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
                    nums = [int(td.get_text(strip=True)) for td in tds
                            if td.get_text(strip=True).isdigit()
                            and 1 <= int(td.get_text(strip=True)) <= 49]
                    if len(nums) == 7:
                        all_results.append(nums)
            time.sleep(0.6)
        except Exception as e:
            print(f"  錯誤: {e}")
    print(f"[大樂透] 共 {len(all_results)} 筆")
    return all_results


def fetch_superlotto(pages=50):  # ← 同樣增加頁數
    all_results = []
    headers = {'User-Agent': 'Mozilla/5.0 Chrome/124.0.0.0 Safari/537.36'}
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
                        if all(1 <= n <= 38 for n in nums[:7]) and 1 <= nums[7] <= 8:
                            all_results.append(nums)
                    except Exception:
                        continue
                elif len(balls) == 0:
                    tds = row.find_all('td')
                    nums = [int(td.get_text(strip=True)) for td in tds
                            if td.get_text(strip=True).isdigit()]
                    if len(nums) == 8:
                        all_results.append(nums)
            time.sleep(0.6)
        except Exception as e:
            print(f"  錯誤: {e}")
    print(f"[威力彩] 共 {len(all_results)} 筆")
    return all_results


def build_features(raw_data, max_num):
    """建立多維特徵：遺漏值 + 頻率 + 近期熱度 + 統計特徵"""
    df = pd.DataFrame(raw_data).iloc[::-1].reset_index(drop=True)

    last_seen  = {i: 0 for i in range(1, max_num + 1)}
    total_freq = {i: 0 for i in range(1, max_num + 1)}
    recent_buf = []  # 最近 20 期

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

        # 近 20 期頻率
        recent_freq = {i: 0 for i in range(1, max_num + 1)}
        for past in recent_buf[-20:]:
            for n in past:
                recent_freq[n] += 1

        # 特徵向量
        feature = (
            list(last_seen.values()) +       # 遺漏值
            list(total_freq.values()) +      # 歷史總頻率
            list(recent_freq.values()) +     # 近期熱度
            [
                sum(curr_set),               # 號碼總和
                sum(1 for n in curr_set if n % 2 == 1),         # 奇數個數
                sum(1 for n in curr_set if n <= max_num // 2),  # 小號個數
                max(curr_set) - min(curr_set),                  # 號碼跨度
            ]
        )
        X_train.append(feature)

        # 標籤
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


def predict(raw_data, max_num, pick):
    if len(raw_data) < 10:
        return [], None

    X_train, y_train, last_seen, total_freq, recent_buf = build_features(raw_data, max_num)

    # Ensemble：RandomForest + ExtraTrees 投票
    rf  = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
    et  = ExtraTreesClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)

    X = np.array(X_train[:-1])
    y = np.array(y_train[1:])

    rf.fit(X, y)
    et.fit(X, y)

    # 最新特徵
    recent_freq = {i: 0 for i in range(1, max_num + 1)}
    for past in recent_buf[-20:]:
        for n in past:
            recent_freq[n] += 1

    latest_x = np.array([[
        *list(last_seen.values()),
        *list(total_freq.values()),
        *list(recent_freq.values()),
        0, 0, 0, 0  # 預測時統計特徵補 0
    ]])

    # 兩個模型機率平均
    def get_probs(model):
        probas = model.predict_proba(latest_x)
        return np.array([p[0][1] if p[0].shape[0] > 1 else 0 for p in probas])

    probs = (get_probs(rf) + get_probs(et)) / 2.0

    top_idx  = np.argsort(probs)[-(pick + 1):][::-1]
    res_nums = sorted([int(i + 1) for i in top_idx])
    return res_nums[:pick], res_nums[-1]


def main():
    now_str = datetime.now(TW_TZ).strftime('%Y-%m-%d %H:%M:%S')
    result  = {"update_time": now_str}

    # 大樂透
    print("\n🎱 抓取大樂透...")
    lotto_data = fetch_lotto649(pages=50)
    nums, special = predict(lotto_data, max_num=49, pick=6)
    result["lotto649"] = {
        "predict_numbers": nums,
        "special_number":  special,
        "status":          "Success" if nums else "Error",
        "data_count":      len(lotto_data)
    }
    print(f"✅ 大樂透：{nums} 特別號：{special}")

    # 威力彩
    print("\n⚡ 抓取威力彩...")
    super_data = fetch_superlotto(pages=50)
    nums, special = predict(super_data, max_num=38, pick=7)
    result["superlotto"] = {
        "predict_numbers": nums,
        "special_number":  special,
        "status":          "Success" if nums else "Error",
        "data_count":      len(super_data)
    }
    print(f"✅ 威力彩：{nums} 第二區：{special}")

    with open("result.json", "w", encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"\n✅ 完成！")


main()
