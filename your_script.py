import pandas as pd
import numpy as np
import requests
import json
import io
import time
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import pytz

# 設定台灣時區
TW_TZ = pytz.timezone('Asia/Taipei')

# ─────────────────────────────────────────
# 1. 備援函數：爬取台彩新版 API (JSON)
# ─────────────────────────────────────────
def fetch_history_api_backup():
    """
    當 CSV 下載失敗或資料不足時，嘗試從台彩 API 抓取當月資料。
    """
    all_results = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/javascript, */*',
        'X-Requested-With': 'XMLHttpRequest',
        'Referer': 'https://www.taiwanlottery.com/lotto/history/history_result/'
    }
    
    now = datetime.now(TW_TZ)
    roc_year = now.year - 1911
    # 嘗試抓取本月與上月的資料
    months = [now.month, (now.month - 1) if now.month > 1 else 12]
    
    for m in months:
        url = f"https://www.taiwanlottery.com/api/Lotto649/GetLotto649HistoryResult?year={roc_year}&month={m}"
        try:
            res = requests.get(url, headers=headers, timeout=20)
            if res.status_code == 200:
                data = res.json()
                # 解析台彩 API 結構
                rows = data.get('data', [])
                for item in rows:
                    try:
                        # 官方 API 欄位通常為 n1~n6 與 sNo (特別號)
                        nums = [int(item[f'n{i}']) for i in range(1, 7)]
                        nums.append(int(item['sNo']))
                        if all(1 <= n <= 49 for n in nums):
                            all_results.append(nums)
                    except: continue
            time.sleep(1)
        except Exception as e:
            print(f"⚠️ API 備援抓取月份 {m} 失敗: {e}")
            
    return all_results

# ─────────────────────────────────────────
# 2. 主函數：下載官方歷史 CSV
# ─────────────────────────────────────────
def fetch_official_csv():
    now = datetime.now(TW_TZ)
    roc_year = now.year - 1911
    all_results = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Referer': 'https://www.taiwanlottery.com/'
    }

    # 抓取今年與去年的資料以確保訓練樣本充足
    for y in [roc_year, roc_year - 1]:
        url = f"https://www.taiwanlottery.com/lotto/history/result_download/download?year={y}"
        try:
            res = requests.get(url, headers=headers, timeout=30)
            if res.status_code != 200:
                print(f"⚠️ {y}年 CSV 無法下載 (Status: {res.status_code})")
                continue

            # 嘗試不同編碼解析 CSV
            text = ""
            for enc in ['utf-8-sig', 'big5', 'cp950']:
                try:
                    text = res.content.decode(enc)
                    break
                except: continue
            
            if not text: continue

            # 解析 CSV，跳過標題雜訊
            df_raw = pd.read_csv(io.StringIO(text), header=None)
            # 篩選出「大樂透」字樣的列
            lotto_rows = df_raw[df_raw.iloc[:, 0].astype(str).str.contains('大樂透', na=False)]
            
            for _, row in lotto_rows.iterrows():
                try:
                    # 獎號通常在 6~12 欄 (包含特別號)
                    nums = [int(row.iloc[i]) for i in range(6, 13) if str(row.iloc[i]).strip().isdigit()]
                    if len(nums) == 7:
                        all_results.append(nums)
                except: continue
            print(f"✅ {y}年 CSV 解析成功，目前累計 {len(all_results)} 筆")
        except Exception as e:
            print(f"❌ {y}年 CSV 處理失敗: {e}")
            
    return all_results

# ─────────────────────────────────────────
# 3. 執行主程式
# ─────────────────────────────────────────
if __name__ == "__main__":
    print(f"🚀 啟動預測任務... 執行時間: {datetime.now(TW_TZ).strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 第一步：抓取資料
        raw_data = fetch_official_csv()
        
        # 如果 CSV 資料不足（例如官方還沒更新本月），則啟動 API 備援
        if len(raw_data) < 20:
            print("💡 CSV 數據不足，啟動 API 備援方案...")
            raw_data += fetch_history_api_backup()

        if len(raw_data) < 10:
            raise ValueError(f"抓取到的資料太少 ({len(raw_data)} 筆)，無法進行 AI 訓練")

        # 第二步：資料預處理 (由舊到新)
        df = pd.DataFrame(raw_data).iloc[::-1].reset_index(drop=True)
        
        # 建立遺漏值特徵
        last_seen = {i: 0 for i in range(1, 50)}
        X_train, y_train = [], []
        
        for _, row in df.iterrows():
            X_train.append(list(last_seen.values()))
            curr_set = set(int(v) for v in row if 1 <= int(v) <= 49)
            # 更新下一期的遺漏值
            for i in range(1, 50):
                last_seen[i] = 0 if i in curr_set else last_seen[i] + 1
            
            # 建立多標籤標記 (Multi-label)
            label = np.zeros(49)
            for n in curr_set:
                label[n-1] = 1
            y_train.append(label)

        # 第三步：AI 模型訓練 (RandomForest)
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        # 用前 N-1 期的遺漏值預測第 N 期的中獎號碼
        model.fit(np.array(X_train[:-1]), np.array(y_train[1:]))

        # 第四步：執行預測
        latest_x = np.array([list(last_seen.values())])
        all_probs_list = model.predict_proba(latest_x)
        
        # 安全取得各號碼的機率 (處理機率陣列結構)
        final_probs = []
        for p in all_probs_list:
            prob_val = p[0][1] if p[0].shape[0] > 1 else 0
            final_probs.append(prob_val)
        
        # 取得機率最高的 7 個號碼 (6 普 + 1 特)
        top_indices = np.argsort(final_probs)[-7:][::-1]
        predicted_all = sorted([int(i + 1) for i in top_indices])

        # 第五步：輸出 JSON 結果
        output = {
            "update_time": datetime.now(TW_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            "predict_numbers": predicted_all[:6],
            "special_number": predicted_all[-1],
            "data_count": len(raw_data),
            "status": "Success"
        }
        
        with open("result.json", "w", encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
            
        print(f"✅ 預測成功！結果已存至 result.json")

    except Exception as e:
        # 即使出錯也產出含有 Error 訊息的 JSON，避免 App 解析失敗
        error_output = {
            "update_time": datetime.now(TW_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            "status": f"Error: {str(e)}",
            "predict_numbers": [],
            "special_number": None
        }
        with open("result.json", "w", encoding='utf-8') as f:
            json.dump(error_output, f, indent=4, ensure_ascii=False)
        print(f"❌ 執行發生錯誤: {e}")
