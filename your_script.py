import json
import random
from datetime import datetime

# 模擬預測（之後你可以換成 AI）
numbers = sorted(random.sample(range(1, 50), 6))

result = {
    "time": datetime.utcnow().isoformat(),
    "prediction": numbers
}

# 輸出 JSON（GitHub Actions 會 commit 這個）
with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("Prediction generated:", result)
