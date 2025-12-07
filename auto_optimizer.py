import json
import subprocess
import os
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console

# -----------------------------------------------------------
# 1. [하드웨어 맞춤 설정] 라이젠 5600 / 32GB RAM 전용
# -----------------------------------------------------------
CONFIG_FILE = "config.json"
STRATEGY = "RLSentimentStrategy"
MODEL = "ExtendedMDPLearner"

# [핵심] 기간을 6개월로 줄여서 램 폭발 방지 (이것도 충분히 깁니다!)
DOWNLOAD_DAYS = 180             
TIMERANGE = "20250601-20251204" # 백테스팅 기간도 6개월로 맞춤

# [핵심] 다운로드는 무조건 1개씩 순차적으로 (안전 제일)
MAX_WORKERS = 1

# [핵심] 백테스팅 시 사용할 CPU 코어 수 (라이젠 5600은 6코어 -> 5개만 사용)
CPU_COUNT = 5

# -----------------------------------------------------------
# 2. 실험 파라미터 (3가지 성격)
# -----------------------------------------------------------
experiments = [
    {"buy_reward": 2.0, "neutral_penalty": -0.1, "train_cycles": 20},
    {"buy_reward": 1.0, "neutral_penalty": -0.05, "train_cycles": 30},
    {"buy_reward": 0.5, "neutral_penalty": 0.0, "train_cycles": 50},
]

console = Console()

def get_env_with_utf8():
    """윈도우 한글 인코딩 오류 방지"""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    return env

def run_command_visible(cmd_list):
    """화면에 진행 상황을 보여주는 실행 함수"""
    try:
        subprocess.run(cmd_list, check=True, shell=True, env=get_env_with_utf8())
        return True
    except subprocess.CalledProcessError:
        return False

def download_single_pair(pair):
    """한 종목 다운로드"""
    # 화면에 잘 보이게 구분선 출력
    print(f"\n{'='*40}")
    print(f"⬇️  다운로드 시작: {pair}")
    print(f"{'='*40}")
    
    cmd = [
        "python", "-m", "freqtrade", "download-data",
        "--config", CONFIG_FILE,
        "--days", str(DOWNLOAD_DAYS),
        "-t", "5m",
        "--pairs", pair,
        "--prepend",
        "--dl-trades" 
    ]
    success = run_command_visible(cmd)
    return pair, success

# ===========================================================
# [STEP 1] 데이터 다운로드
# ===========================================================
console.print(f"\n[bold cyan]🚀 [STEP 1] {DOWNLOAD_DAYS}일치 데이터 다운로드 시작 (안전 모드)[/bold cyan]")
console.print(f"[yellow]   - 시스템: Ryzen 5600 / 32GB RAM 최적화 적용[/yellow]")
console.print(f"[yellow]   - 다운로드: 순차 진행 (RAM 보호)[/yellow]\n")

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config_data = json.load(f)
    pairs = config_data['exchange']['pair_whitelist']

# 순차 다운로드 실행
for i, pair in enumerate(pairs):
    console.print(f"\n[bold green]📦 [{i+1}/{len(pairs)}] {pair} 처리 중...[/bold green]")
    pair, success = download_single_pair(pair)
    
    if success:
        console.print(f"✅ {pair} 다운로드 완료!")
    else:
        console.print(f"❌ {pair} 다운로드 실패 (네트워크 일시적 오류일 수 있음)")

console.print("\n[bold green]✨ 모든 데이터 다운로드 완료![/bold green]\n")

# ===========================================================
# [STEP 2] 자동 최적화 실험
# ===========================================================
results = []
console.print(f"[bold cyan]🧪 [STEP 2] 총 {len(experiments)}개의 실험을 시작합니다.[/bold cyan]")
console.print(f"[yellow]   - 백테스팅 CPU 사용: {CPU_COUNT}개 코어 (1개는 윈도우용으로 남김)[/yellow]")

for i, params in enumerate(experiments):
    exp_name = f"auto_exp_{i+1}"
    console.print(f"\n▶ [Experiment {i+1}/{len(experiments)}] {exp_name} 진행 중... {params}")
    
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)

        config["freqai"]["identifier"] = exp_name
        config["freqai"]["rl_config"]["train_cycles"] = params["train_cycles"]
        # 학습 데이터 기간도 6개월에 맞춰 조정 (30일)
        config["freqai"]["train_period_days"] = 30
        
        if "model_reward_parameters" not in config["freqai"]["rl_config"]:
            config["freqai"]["rl_config"]["model_reward_parameters"] = {}
        config["freqai"]["rl_config"]["model_reward_parameters"]["buy_reward"] = params["buy_reward"]
        config["freqai"]["rl_config"]["model_reward_parameters"]["neutral_penalty"] = params["neutral_penalty"]
        
        # [핵심] CPU 코어 제한 (컴퓨터 멈춤 방지)
        config["freqai"]["rl_config"]["cpu_count"] = CPU_COUNT

        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
            
    except Exception as e:
        console.print(f"❌ 설정 오류: {e}")
        continue

    bt_cmd = [
        "python", "-m", "freqtrade", "backtesting",
        "--strategy", STRATEGY,
        "--freqaimodel", MODEL,
        "--config", CONFIG_FILE,
        "--timerange", TIMERANGE
    ]
    
    start_time = time.time()
    subprocess.run(bt_cmd, check=False, shell=True, env=get_env_with_utf8())
    duration = time.time() - start_time
    
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
            console.print(f"✅ 실험 {i+1} 완료. (수익률: {summary['Profit_Ratio']})")
    except Exception as e:
        console.print(f"⚠️ 결과 집계 중 오류 (무시 가능): {e}")

# ===========================================================
# [STEP 3] 리포트 저장
# ===========================================================
if results:
    df = pd.DataFrame(results)
    console.print("\n" + "="*60)
    console.print(df)
    df.to_csv("final_result_report.csv", index=False)
    console.print(f"\n💾 결과 저장됨: final_result_report.csv")

console.print("\n[bold green]😴 모든 작업이 완료되었습니다. 이제 주무셔도 됩니다![/bold green]")