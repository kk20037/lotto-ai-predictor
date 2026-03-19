import pandas as pd
import numpy as np
import requests
import json
import io
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import pytz
import time

TW_TZ = pytz.timezone('Asia/Taipei')

def fetch_official_csv():
    now = datetime.now(TW_TZ)
    roc_year = now.year - 1911
    all_results = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 'Referer': 'https://www.taiwanlottery.com/'}

    for y in [roc_year, roc_year - 1]:
        url = f"https://www.taiwanlottery.com/lotto/history/result_download/download?year={y}"
        try:
            res = requests.get(url, headers=headers, timeout=30)
            if res.status_code != 200: continue

            for enc in ['utf-8-sig', 'big5', 'cp950']:
                try:
                    text = res.content.decode(enc)
                    break
                except: continue
            
            # 強健解析：跳過 CSV 標題雜訊
            df_raw = pd.read_csv(io.StringIO(text), header=None)
            # 找到大樂透資料列
            lotto_rows = df_raw[df_raw.iloc[:, 0].astype(str).str.contains('大樂透', na=False)]
            
            for _, row in lotto_rows.iterrows():
                try:
                    # 台彩 CSV 格式：獎號通常在 6~11 欄，特別號在 12 欄
                    nums = [int(row.iloc[i]) for i in range(6, 13) if str(row.iloc[i]).strip().isdigit()]
                    if len(nums) == 7:
                        all_results.append(nums)
                except: continue
            print(f"✅ {y}年 解析成功，累計 {len(all_results)} 筆")
        except Exception as e:
            print(f"❌ {y}年 讀取失敗: {e}")
    return all_results

# ... (fetch_history_page_backup 保持你原本的寫法) ...

# 主執行邏輯
try:
    print("🔍 啟動抓取任務...")
    raw_data = fetch_official_csv()
    # 如果 CSV 失敗，嘗試 API
    if len(raw_data) < 20:
        from fetch_history_page_backup import fetch_history_page_backup # 假設你在同個檔
        raw_data += fetch_history_page_backup()

    if len(raw_data) < 10:
        raise ValueError("資料抓取量不足以訓練 AI")

    # AI 訓練邏輯 (加上防錯機率提取)
    df = pd.DataFrame(raw_data).iloc[::-1].reset_index(drop=True)
    last_seen = {i: 0 for i in range(1, 50)}
    X_train, y_train = [], []
    for _, row in df.iterrows():
        X_train.append(list(last_seen.values()))
        curr = set(int(v) for v in row if 1 <= int(v) <= 49)
        for i in range(1, 50):
            last_seen[i] = 0 if i in curr else last_seen[i] + 1
        label = np.zeros(49)
        for n in curr: label[n-1] = 1
        y_train.append(label)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(np.array(X_train[:-1]), np.array(y_train[1:]))

    latest_x = np.array([list(last_seen.values())])
    all_probs = model.predict_proba(latest_x)
    probs = np.array([p[0][1] if p[0].shape[0] > 1 else 0 for p in all_probs])

    top_idx = np.argsort(probs)[-7:][::-1]
    res_nums = sorted(int(i + 1) for i in top_idx)

    # 輸出結果
    result = {
        "update_time": datetime.now(TW_TZ).strftime('%Y-%m-%d %H:%M:%S'),
        "predict_numbers": res_nums[:6],
        "special_number": res_nums[-1],
        "status": "Success",
        "data_count": len(raw_data)
    }
    with open("result.json", "w", encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print("🚀 預測成功並更新 JSON")

except Exception as e:
    # 這裡確保即使出錯也會產出 result.json，避免 Android 端閃退
    error_res = {
        "update_time": datetime.now(TW_TZ).strftime('%Y-%m-%d %H:%M:%S'),
        "status": f"Error: {str(e)}"
    }
    with open("result.json", "w", encoding='utf-8') as f:
        json.dump(error_res, f, indent=4)
    print(f"❌ 執行中斷: {e}")
