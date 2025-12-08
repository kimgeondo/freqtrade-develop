# Force Push 실행 가이드

## ⚠️ 중요 알림

저는 자동화 시스템의 제약으로 인해 force push를 직접 실행할 수 없습니다.
아래 명령어를 로컬에서 직접 실행해주세요.

## Force Push 실행 방법

### 1단계: 로컬에서 리포지토리 준비

```bash
# 현재 디렉토리로 이동 (리포지토리가 없다면)
git clone https://github.com/kimgeondo/freqtrade-develop.git
cd freqtrade-develop

# 브랜치 체크아웃
git checkout copilot/reduce-file-size-under-100mb

# 최신 상태로 업데이트
git pull origin copilot/reduce-file-size-under-100mb
```

### 2단계: Git History 정리

```bash
# git-filter-repo 설치 (아직 없다면)
pip install git-filter-repo

# 삭제할 파일 목록 생성
cat > /tmp/paths-to-remove.txt << 'EOF'
user_data/models/long_term_exp_1/
user_data/models/long_term_exp_2/
user_data/models/auto_exp_1/
user_data/models/auto_exp_2/
user_data/models/auto_exp_3/
user_data/models/grad_project_debug/
user_data/models/grad_project_final/
user_data/models/grad_project_real_final/
user_data/models/grad_project_success/
user_data/data/binance/
build_helpers/pyarrow-22.0.0-cp311-cp311-linux_armv7l.whl
build_helpers/ta_lib-0.6.8-cp311-cp311-manylinux_2_31_armv7l.whl
build_helpers/ta_lib-0.6.8-cp313-cp313-manylinux_2_31_armv7l.whl
tests/testdata/orderflow/populate_dataframe_with_trades_TRADES.feather
EOF

# git history에서 파일 제거
git filter-repo --invert-paths --paths-from-file /tmp/paths-to-remove.txt --force
```

### 3단계: Remote 재설정 및 Force Push

```bash
# remote 다시 추가 (filter-repo가 제거함)
git remote add origin https://github.com/kimgeondo/freqtrade-develop.git

# Force Push 실행
git push --force origin copilot/reduce-file-size-under-100mb
```

### 4단계: 크기 확인

```bash
# 최종 크기 확인
du -sh .
du -sh .git

# 예상 결과:
# 전체: ~45MB
# .git: ~15MB
```

## 또는 더 간단한 방법 (BFG Repo Cleaner 사용)

```bash
# BFG 다운로드
wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar

# 백업 생성
git clone --mirror https://github.com/kimgeondo/freqtrade-develop.git freqtrade-develop-mirror
cd freqtrade-develop-mirror

# 대용량 파일 제거
java -jar ../bfg-1.14.0.jar --strip-blobs-bigger-than 10M .

# 정리
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Push
git push --force
```

## ⚠️ 팀원들에게 공지사항

Force push 후 모든 팀원은 다음을 실행해야 합니다:

```bash
# 방법 1: 새로 클론
rm -rf freqtrade-develop
git clone https://github.com/kimgeondo/freqtrade-develop.git

# 방법 2: 강제 리셋
cd freqtrade-develop
git fetch origin
git reset --hard origin/copilot/reduce-file-size-under-100mb
```

## 예상 결과

- **이전**: 465MB (추적 파일 + .git 히스토리)
- **이후**: 45MB (추적 파일 30MB + .git 15MB)
- **감소**: 420MB (90% 축소)

## 문제 해결

### "refusing to delete the current branch" 에러
```bash
git checkout main
git push --force origin copilot/reduce-file-size-under-100mb
```

### "permission denied" 에러
- GitHub Settings → Repository → Push access 확인
- Personal Access Token 권한 확인

### 다른 문제가 있다면
1. 에러 메시지 전체를 복사
2. 실행한 명령어와 함께 알려주세요

---
*작성: @copilot*
*날짜: 2025-12-08*
