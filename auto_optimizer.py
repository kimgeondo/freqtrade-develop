import json
import subprocess
import os
import pandas as pd
import time

# -----------------------------------------------------------
# 1. 기본 설정
# -----------------------------------------------------------
CONFIG_FILE = "config.json"
STRATEGY = "RLSentimentStrategy"
MODEL = "ExtendedMDPLearner"
TIMERANGE = "20240101-20251204"
DOWNLOAD_DAYS = 730             

# -----------------------------------------------------------
# 2. 실험 파라미터
# -----------------------------------------------------------
experiments = [
    {"buy_reward": 2.0, "neutral_penalty": -0.1, "train_cycles": 20},
    {"buy_reward": 1.0, "neutral_penalty": -0.05, "train_cycles": 30},
    {"buy_reward": 0.5, "neutral_penalty": 0.0, "train_cycles": 50},
]

def run_command_visible(cmd_list):
    """화면에 진행 상황을 그대로 보여주는 실행 함수"""
    try:
        # stdout, stderr를 캡처하지 않고 그대로 내보냄 -> 진행바가 보임!
        subprocess.run(cmd_list, check=True, shell=True)
        return True
    except subprocess.CalledProcessError:
        return False

# ===========================================================
# [STEP 1] 데이터 다운로드 (순차 진행 & 화면 표시)
# ===========================================================
print("\n" + "="*60)
print(f"🚀 [STEP 1] {DOWNLOAD_DAYS}일치 데이터 다운로드 시작 (순차 진행)")
print("   (화면에 Freqtrade 진행바가 표시됩니다.)")
print("="*60)

# 종목 리스트 가져오기
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config_data = json.load(f)
    pairs = config_data['exchange']['pair_whitelist']

# 한 종목씩 차례대로 다운로드 (진행바를 보기 위해)
for i, pair in enumerate(pairs):
    print(f"\n⬇️ [{i+1}/{len(pairs)}] 다운로드 중: {pair}")
    
    dl_cmd = [
        "python", "-m", "freqtrade", "download-data",
        "--config", CONFIG_FILE,
        "--days", str(DOWNLOAD_DAYS),
        "-t", "5m",
        "--pairs", pair,
        "--prepend",
        "--dl-trades" 
    ]
    
    success = run_command_visible(dl_cmd)
    
    if success:
        print(f"✅ {pair} 완료!")
    else:
        print(f"❌ {pair} 실패 (네트워크/거래소 오류 가능성)")

print("\n✨ 모든 데이터 다운로드 절차가 끝났습니다!\n")

# ===========================================================
# [STEP 2] 자동 최적화 실험
# ===========================================================
results = []
print("="*60)
print(f"🧪 [STEP 2] 총 {len(experiments)}개의 실험을 시작합니다.")
print("="*60)

for i, params in enumerate(experiments):
    exp_name = f"auto_exp_{i+1}"
    print(f"\n▶ [Experiment {i+1}/{len(experiments)}] {exp_name} 진행 중... {params}")
    
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)

        config["freqai"]["identifier"] = exp_name
        config["freqai"]["rl_config"]["train_cycles"] = params["train_cycles"]
        config["freqai"]["train_period_days"] = 45
        
        if "model_reward_parameters" not in config["freqai"]["rl_config"]:
            config["freqai"]["rl_config"]["model_reward_parameters"] = {}
        config["freqai"]["rl_config"]["model_reward_parameters"]["buy_reward"] = params["buy_reward"]
        config["freqai"]["rl_config"]["model_reward_parameters"]["neutral_penalty"] = params["neutral_penalty"]
        
        # 안전장치: CPU 코어 제한 (메모리 보호)
        config["freqai"]["rl_config"]["cpu_count"] = 4

        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
            
    except Exception as e:
        print(f"❌ 설정 오류: {e}")
        continue

    bt_cmd = [
        "python", "-m", "freqtrade", "backtesting",
        "--strategy", STRATEGY,
        "--freqaimodel", MODEL,
        "--config", CONFIG_FILE,
        "--timerange", TIMERANGE
    ]
    
    start_time = time.time()
    # 백테스팅 진행 상황도 화면에 보이게 설정
    run_command_visible(bt_cmd)
    duration = time.time() - start_time
    
    # 결과 파싱
    try:
        result_dir = "user_data/backtest_results"
        files = [os.path.join(result_dir, f) for f in os.listdir(result_dir) if f.endswith(".json") and not f.endswith(".meta.json")]
        if files:
            latest_file = max(files, key=os.path.getctime)
            with open(latest_file, "r") as f:
                res_data = json.load(f)
            
            strat_res = res_data["strategy"][STRATEGY]
            summary = {
                "Experiment": exp_name,
                "Buy_Reward": params["buy_reward"],
                "Trades": strat_res["total_trades"],
                "Win_Rate": f"{strat_res['win_rate'] * 100:.2f}%",
                "Profit_Ratio": f"{strat_res['profit_total_ratio'] * 100:.2f}%",
                "Profit_USDT": f"{strat_res['profit_total']:.2f}",
                "Drawdown": f"{strat_res['max_drawdown_account'] * 100:.2f}%"
            }
            results.append(summary)
            print(f"✅ 실험 {i+1} 완료. (수익률: {summary['Profit_Ratio']})")
    except Exception as e:
        print(f"⚠️ 결과 집계 중 오류 (무시 가능): {e}")

# ===========================================================
# [STEP 3] 리포트 저장
# ===========================================================
if results:
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print(df)
    df.to_csv("final_result_report.csv", index=False)
    print(f"\n💾 결과 저장됨: final_result_report.csv")

print("\n😴 모든 작업이 완료되었습니다. 이제 주무셔도 됩니다!")