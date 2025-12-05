import json
import subprocess
import os
import pandas as pd
import time

# -----------------------------------------------------------
# 1. [핵심 변경] 기간 설정 (2년치 데이터)
# -----------------------------------------------------------
CONFIG_FILE = "config.json"
STRATEGY = "RLSentimentStrategy"
MODEL = "ExtendedMDPLearner"

# 2년치 데이터 다운로드 (730일)
DOWNLOAD_DAYS = 730             

# 백테스팅 기간: 2024년 1월 1일 ~ 2025년 12월 4일 (약 2년)
# (데이터가 충분히 확보된 구간으로 설정)
TIMERANGE = "20240101-20251204"

# -----------------------------------------------------------
# 2. 실험할 파라미터 조합 (최적의 보상 찾기)
# -----------------------------------------------------------
experiments = [
    # 실험 A: 공격형 (매수 보상 2.0 / 관망 벌점 -0.1)
    # -> 상승장에서 유리할 것으로 예상
    {"buy_reward": 2.0, "neutral_penalty": -0.1, "train_cycles": 20},
    
    # 실험 B: 밸런스형 (매수 보상 1.0 / 관망 벌점 -0.05)
    # -> 하락장/횡보장에서 방어력이 좋을 것으로 예상
    {"buy_reward": 1.0, "neutral_penalty": -0.05, "train_cycles": 25},
]

def run_command(cmd_list):
    """명령어 실행 및 에러 처리 함수"""
    print(f"\n[EXEC] {' '.join(cmd_list)}")
    try:
        subprocess.run(cmd_list, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 오류 발생! (에러 코드: {e.returncode})")
        # 치명적이지 않은 에러면 계속 진행 (예: 일부 데이터 누락)
        pass

# ===========================================================
# [STEP 1] 대용량 데이터 다운로드 (2년치)
# ===========================================================
print("="*60)
print(f"🚀 [STEP 1] {DOWNLOAD_DAYS}일치 데이터 다운로드 시작...")
print("   (시간이 조금 걸릴 수 있습니다. 잠시만 기다려주세요.)")
print("="*60)

dl_cmd = [
    "python", "-m", "freqtrade", "download-data",
    "--config", CONFIG_FILE,
    "--days", str(DOWNLOAD_DAYS),
    "-t", "5m",
    "--prepend"
]
run_command(dl_cmd)
print("✅ 데이터 다운로드 완료!\n")

# ===========================================================
# [STEP 2] 장기 백테스팅 실험 루프
# ===========================================================
results = []
print("="*60)
print(f"🧪 [STEP 2] 총 {len(experiments)}개의 장기 실험을 시작합니다.")
print("   (2년치 시뮬레이션이므로 실험당 10~30분 소요될 수 있습니다.)")
print("="*60)

for i, params in enumerate(experiments):
    exp_name = f"long_term_exp_{i+1}"
    print(f"\n▶ [Experiment {i+1}/{len(experiments)}] {exp_name} 진행 중... {params}")
    
    # 1. config.json 수정
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 파라미터 주입
        config["freqai"]["identifier"] = exp_name
        config["freqai"]["rl_config"]["train_cycles"] = params["train_cycles"]
        
        if "model_reward_parameters" not in config["freqai"]["rl_config"]:
            config["freqai"]["rl_config"]["model_reward_parameters"] = {}
            
        config["freqai"]["rl_config"]["model_reward_parameters"]["buy_reward"] = params["buy_reward"]
        config["freqai"]["rl_config"]["model_reward_parameters"]["neutral_penalty"] = params["neutral_penalty"]

        # [중요] 학습 데이터 기간도 살짝 늘려줌 (30일 -> 45일)
        config["freqai"]["train_period_days"] = 45

        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
            
    except Exception as e:
        print(f"❌ 설정 파일 수정 중 오류: {e}")
        continue

    # 2. 백테스팅 실행
    bt_cmd = [
        "python", "-m", "freqtrade", "backtesting",
        "--strategy", STRATEGY,
        "--freqaimodel", MODEL,
        "--config", CONFIG_FILE,
        "--timerange", TIMERANGE
    ]
    
    start_time = time.time()
    run_command(bt_cmd)
    duration = time.time() - start_time
    
    # 3. 결과 파싱
    try:
        result_dir = "user_data/backtest_results"
        files = [os.path.join(result_dir, f) for f in os.listdir(result_dir) if f.endswith(".json") and not f.endswith(".meta.json")]
        if not files:
            print("⚠️ 결과 파일을 찾을 수 없습니다.")
            continue
            
        latest_file = max(files, key=os.path.getctime)
        
        with open(latest_file, "r") as f:
            res_data = json.load(f)
            
        strat_res = res_data["strategy"][STRATEGY]
        
        summary = {
            "Experiment": exp_name,
            "Buy_Reward": params["buy_reward"],
            "Neutral_Penalty": params["neutral_penalty"],
            "Trades": strat_res["total_trades"],
            "Win_Rate": f"{strat_res['win_rate'] * 100:.2f}%",
            "Profit_Ratio": f"{strat_res['profit_total_ratio'] * 100:.2f}%",
            "Profit_USDT": f"{strat_res['profit_total']:.2f}",
            "Max_Drawdown": f"{strat_res['max_drawdown_account'] * 100:.2f}%",
            "Duration_Min": int(duration / 60)
        }
        results.append(summary)
        print(f"✅ 실험 {i+1} 성공! (수익: {summary['Profit_USDT']} USDT, 거래수: {summary['Trades']})")

    except Exception as e:
        print(f"⚠️ 결과 파싱 중 오류: {e}")

# ===========================================================
# [STEP 3] 최종 리포트
# ===========================================================
if results:
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("🏆 2년 장기 백테스팅 최종 결과")
    print("="*60)
    print(df)
    
    csv_filename = "long_term_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n💾 결과 저장 완료: {csv_filename}")
else:
    print("\n❌ 저장된 결과가 없습니다.")

print("\n😴 2년치 테스트가 모두 끝났습니다. 수고하셨습니다!")