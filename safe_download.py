import json
import subprocess
import os
from datetime import datetime, timedelta

# ==========================================
# 사용자 설정 (여기만 수정하세요)
# ==========================================
CONFIG_FILE = "config.json"
TOTAL_DAYS = 180      # 전체 받을 기간 (예: 180일)
CHUNK_DAYS = 20       # [핵심] 한 번에 받을 기간 (20일씩 끊어서 받음 -> 램 보호!)
TIMEFRAME = "5m"      # 시간봉
# ==========================================

def get_env_with_utf8():
    """윈도우 한글 인코딩 오류 방지용 환경변수 설정"""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    return env

def run_download():
    # 1. 설정 파일에서 종목 리스트 읽기
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ 오류: {CONFIG_FILE} 파일이 없습니다.")
        return

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
            pairs = config['exchange']['pair_whitelist']
    except Exception as e:
        print(f"❌ 설정 파일 읽기 실패: {e}")
        return

    print(f"\n🚀 [안전 모드] 데이터 분할 다운로드 시작")
    print(f"   - 대상 종목: {len(pairs)}개")
    print(f"   - 전체 기간: {TOTAL_DAYS}일")
    print(f"   - 분할 단위: {CHUNK_DAYS}일씩 끊어서 저장 (RAM 보호)\n")

    # 오늘 날짜 기준 시작일 계산
    end_date = datetime.now()
    start_date = end_date - timedelta(days=TOTAL_DAYS)

    # 2. 종목별로 순회
    for i, pair in enumerate(pairs):
        print(f"==================================================")
        print(f"📦 [{i+1}/{len(pairs)}] 종목 처리 중: {pair}")
        print(f"==================================================")

        # 3. 기간별로 쪼개서 다운로드 (Chunking)
        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=CHUNK_DAYS), end_date)
            
            # 날짜 포맷 변환 (YYYYMMDD)
            timerange = f"{current_start.strftime('%Y%m%d')}-{current_end.strftime('%Y%m%d')}"
            
            print(f"   ⬇️  다운로드 구간: {timerange} ({CHUNK_DAYS}일치)...")

            cmd = [
                "python", "-m", "freqtrade", "download-data",
                "--config", CONFIG_FILE,
                "-t", TIMEFRAME,
                "--pairs", pair,
                "--timerange", timerange,  # [핵심] 쪼개진 기간만 다운로드
                "--dl-trades"              # 오더플로우 데이터 포함
            ]

            try:
                # 실행 및 대기
                subprocess.run(cmd, check=True, shell=True, env=get_env_with_utf8())
            except subprocess.CalledProcessError:
                print(f"   ❌ 구간 실패: {timerange} (다음 구간으로 넘어갑니다)")
            
            # 다음 구간으로 이동
            current_start = current_end

        print(f"   ✅ {pair} 전체 완료!\n")

    print("✨ 모든 다운로드 작업이 안전하게 종료되었습니다.")

if __name__ == "__main__":
    run_download()