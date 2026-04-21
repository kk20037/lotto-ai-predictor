import pandas as pd
import numpy as np
import requests
import json
import time
from sklearn.ensemble import ExtraTreesClassifier
from datetime import datetime
import pytz
import urllib3

# 關閉 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
TW_TZ = pytz.timezone('Asia/Taipei')

def fetch_data(lottery_type, pages=2):
    """
    通用抓取函數
    lottery_type: 'Lotto649' 或 'SuperLotto'
    """
    all_results = []
    now = datetime.now(TW_TZ)
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    
    # 決定 JSON 內部的 Key 名稱
    res_key = "lotto649Res" if lottery_type == "Lotto649" else "superLottoRes"
    url = f"https://api.taiwanlottery.com/TLCAPIWeB/Lottery/{lottery_type}Result"

    for i in range(pages):
        month_str = (now - pd.DateOffset(months=i)).strftime('%Y-%m')
        params = {"month": month_str, "pageNum": 1, "pageSize": 50}
        
        print(f"[{lottery_type}] 正在抓取 {month_str}...")
        try:
            res = requests.get(url, params=params, headers=headers, verify=False, timeout=20)
            if res.status_code == 200:
                data = res.json()
                content = data.get("content", {})
                # 威力彩有時大小寫不一，做個容錯
                results = content.get(res_key) or content.get(res_key.lower()) or []
                for item in results:
                    nums = item.get("drawNumberSize", [])
                    if len(nums) == 7:
                        all_results.append(nums)
            time.sleep(1)
        except Exception as e:
            print(f"  抓取 {month_str} 失敗: {e}")
            
    return all_results

def build_features(raw_data, max_num):
    df = pd.DataFrame(raw_data).iloc[::-1].reset_index(drop=True)
    last_seen = {i: 0 for i in range(1, max_num + 1)}
    total_freq = {i: 0 for i in range(1, max_num + 1)}
    recent_buf = []
    X_train, y_train = [], []

    for _, row in df.iterrows():
        curr_set = set(row)
        recent_freq = {i: 0 for i in range(1, max_num + 1)}
        for past in recent_buf[-20:]:
            for n in past: recent_freq[n] += 1

        num_list = sorted(list(curr_set))
        feature = (
            list(last_seen.values()) + 
            list(total_freq.values()) + 
            list(recent_freq.values()) + 
            [sum(num_list), sum(1 for n in num_list if n % 2 == 1), 
             sum(1 for n in num_list if n <= max_num // 2), max(num_list) - min(num_list)]
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

def predict_lotto(raw_data, max_num, pick):
    if len(raw_data) < 10: return [], None
    X_train, y_train, last_seen, total_freq, recent_buf = build_features(raw_data, max_num)
    
    X = np.array(X_train[:-1])
    y = np.array(y_train[1:])

    model = ExtraTreesClassifier(n_estimators=1000, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X, y)

    recent_freq = {i: 0 for i in range(1, max_num + 1)}
    for past in recent_buf[-20:]:
        for n in past: recent_freq[n] += 1

    latest_x = np.array([[*list(last_seen.values()), *list(total_freq.values()), *list(recent_freq.values()), 0, 0, 0, 0]])
    probas = model.predict_proba(latest_x)
    
    probs = []
    for p in probas:
        val = p[0][1] if (isinstance(p, list) and len(p[0]) > 1) or (hasattr(p, 'shape') and p.shape[1] > 1) else 0
        probs.append(val)
    
    top_idx = np.argsort(probs)[-pick:][::-1]
    res_nums = sorted([int(i + 1) for i in top_idx])
    
    special_num = np.argmax(list(last_seen.values())[:8 if max_num == 38 else 49]) + 1
    return res_nums, int(special_num)

def main():
    now_str = datetime.now(TW_TZ).strftime('%Y-%m-%d %H:%M:%S')
    result = {"update_time": now_str}

    # 大樂透
    lotto_data = fetch_data("Lotto649")
    if len(lotto_data) >= 10:
        nums, spec = predict_lotto(lotto_data, 49, 6)
        result["lotto649"] = {"predict_numbers": nums, "special_number": spec, "status": "Success", "data_count": len(lotto_data)}
    else:
        result["lotto649"] = {"predict_numbers": [], "special_number": None, "status": "Error", "data_count": len(lotto_data)}

    # 威力彩
    super_data = fetch_data("SuperLotto")
    if len(super_data) >= 10:
        nums, spec = predict_lotto(super_data, 38, 6)
        result["superlotto"] = {"predict_numbers": nums, "special_number": spec, "status": "Success", "data_count": len(super_data)}
    else:
        result["superlotto"] = {"predict_numbers": [], "special_number": None, "status": "Error", "data_count": len(super_data)}

    with open("result.json", "w", encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"✅ 分析完成！大樂透: {len(lotto_data)} 筆, 威力彩: {len(super_data)} 筆")

if __name__ == "__main__":
    main()
