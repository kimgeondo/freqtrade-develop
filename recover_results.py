import json
import os
import pandas as pd
from datetime import datetime

# 설정
RESULT_DIR = "user_data/backtest_results"
STRATEGY_NAME = "RLSentimentStrategy"

results = []
print(f"📂 '{RESULT_DIR}' 폴더에서 결과 복구 중...")

# 결과 폴더 뒤지기
try:
    files = [f for f in os.listdir(RESULT_DIR) if f.endswith(".json") and not f.endswith(".meta.json")]
    
    # auto_exp_ 로 시작하는 실험 결과만 찾기
    exp_files = {}
    for f in files:
        # 파일 내용을 살짝 읽어서 실험 이름 확인 (혹은 날짜로 추정)
        with open(os.path.join(RESULT_DIR, f), "r", encoding="utf-8") as file_obj:
            try:
                data = json.load(file_obj)
                # 전략 실행 기록이 있는지 확인
                if "strategy" in data and STRATEGY_NAME in data["strategy"]:
                    # 실험 이름(identifier) 확인 방법이 없으므로 파일 수정 시간으로 정렬
                    timestamp = os.path.getmtime(os.path.join(RESULT_DIR, f))
                    exp_files[f] = (timestamp, data)
            except:
                continue

    # 시간순 정렬
    sorted_files = sorted(exp_files.items(), key=lambda x: x[1][0])

    for filename, (ts, data) in sorted_files:
        strat_res = data["strategy"][STRATEGY_NAME]
        
        # 유의미한 거래가 있었던 결과만 복구
        if strat_res["total_trades"] > 0:
            run_date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            summary = {
                "File": filename,
                "Date": run_date,
                "Trades": strat_res["total_trades"],
                "Win_Rate": f"{strat_res['win_rate'] * 100:.2f}%",
                "Profit_Ratio": f"{strat_res['profit_total_ratio'] * 100:.2f}%",
                "Profit_USDT": f"{strat_res['profit_total']:.2f}",
                "Drawdown": f"{strat_res['max_drawdown_account'] * 100:.2f}%"
            }
            results.append(summary)

    if results:
        df = pd.DataFrame(results)
        print("\n✅ 복구된 실험 결과:")
        print(df)
        df.to_csv("recovered_report.csv", index=False)
        print("\n💾 'recovered_report.csv' 파일로 저장했습니다.")
    else:
        print("\n❌ 복구할 유의미한 결과가 없습니다. (모두 0건 거래였거나 파일 없음)")

except Exception as e:
    print(f"ERROR: {e}")