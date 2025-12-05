import json
import subprocess
import os
import pandas as pd
import time

# -----------------------------------------------------------
# 1. 기본 설정 (여기만 확인하세요!)
# -----------------------------------------------------------
CONFIG_FILE = "config.json"
STRATEGY = "RLSentimentStrategy"
MODEL = "ExtendedMDPLearner"
TIMERANGE = "20251128-20251204"  # 검증(백테스팅) 기간
DOWNLOAD_DAYS = 180             # 다운로드할 데이터 기간 (일)

# -----------------------------------------------------------
# 2. 실험할 파라미터 조합 (보상 체계 실험)
# -----------------------------------------------------------
experiments = [
    # 실험 1: 공격형 (매수 +2.0, 관망 -0.1, 짧게 학습)
    {"buy_reward": 2.0, "neutral_penalty": -0.1, "train_cycles": 20},
    
    # 실험 2: 밸런스형 (매수 +1.0, 관망 -0.05, 적당히 학습)
    {"buy_reward": 1.0, "neutral_penalty": -0.05, "train_cycles": 30},
    
    # 실험 3: 신중형 (매수 +0.5, 관망 0.0, 길게 학습)
    {"buy_reward": 0.5, "neutral_penalty": 0.0, "train_cycles": 50},
]

def run_command(cmd_list):
    """명령어를 실행하고 에러 발생 시 중단하는 함수"""
    print(f"\n[EXEC] {' '.join(cmd_list)}")
    try:
        subprocess.run(cmd_list, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 오류 발생! 스크립트를 중단합니다. 에러 코드: {e.returncode}")
        exit(1)

# ===========================================================
# [STEP 1] 데이터 자동 다운로드
# ===========================================================
print("="*50)
print(f"🚀 자동화 프로세스 시작: {DOWNLOAD_DAYS}일치 데이터 다운로드")
print("="*50)

# 다운로드 명령어 (기존 데이터 앞에 붙이기 --prepend 사용)
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
# [STEP 2] 자동 최적화 실험 루프
# ===========================================================
results = []
print("="*50)
print(f"🧪 총 {len(experiments)}개의 실험을 순차적으로 진행합니다.")
print("="*50)

for i, params in enumerate(experiments):
    exp_name = f"auto_exp_{i+1}"
    print(f"\n▶ [Experiment {i+1}/{len(experiments)}] {exp_name} 시작... {params}")
    
    # 1. config.json 수정
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 파라미터 주입
        config["freqai"]["identifier"] = exp_name # 캐시 충돌 방지용 새 이름
        config["freqai"]["rl_config"]["train_cycles"] = params["train_cycles"]
        
        # 보상 파라미터가 없으면 생성
        if "model_reward_parameters" not in config["freqai"]["rl_config"]:
            config["freqai"]["rl_config"]["model_reward_parameters"] = {}
            
        config["freqai"]["rl_config"]["model_reward_parameters"]["buy_reward"] = params["buy_reward"]
        config["freqai"]["rl_config"]["model_reward_parameters"]["neutral_penalty"] = params["neutral_penalty"]

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
    
    # 학습 및 백테스팅 시작
    start_time = time.time()
    run_command(bt_cmd)
    duration = time.time() - start_time
    
    # 3. 결과 파싱 및 저장
    try:
        result_dir = "user_data/backtest_results"
        # 방금 생성된 가장 최신 json 파일 찾기
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
            "Train_Cycles": params["train_cycles"],
            "Trades": strat_res["total_trades"],
            "Win_Rate": f"{strat_res['win_rate'] * 100:.2f}%",
            "Profit_Ratio": f"{strat_res['profit_total_ratio'] * 100:.2f}%",
            "Profit_USDT": f"{strat_res['profit_total']:.2f}",
            "Duration_Sec": int(duration)
        }
        results.append(summary)
        print(f"✅ 실험 {i+1} 성공! (수익률: {summary['Profit_Ratio']}, 거래수: {summary['Trades']})")

    except Exception as e:
        print(f"⚠️ 결과 파싱 중 오류 발생: {e}")

# ===========================================================
# [STEP 3] 최종 리포트 출력 및 저장
# ===========================================================
if results:
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("🏆 최종 실험 결과 리포트")
    print("="*60)
    print(df)
    
    # CSV 저장
    csv_filename = "final_experiment_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n💾 결과가 '{csv_filename}' 파일로 저장되었습니다.")
else:
    print("\n❌ 저장된 결과가 없습니다.")

print("\n😴 모든 작업이 완료되었습니다. 이제 주무셔도 됩니다!")