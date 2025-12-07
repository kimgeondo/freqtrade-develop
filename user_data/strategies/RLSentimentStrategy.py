import logging
from functools import reduce
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
import requests
from datetime import datetime, timedelta

from freqtrade.strategy import IStrategy
from freqtrade.enums import RunMode  # 실행 모드 확인용

logger = logging.getLogger(__name__)

class RLSentimentStrategy(IStrategy):
    
    minimal_roi = {"0": 0.1, "20": 0.05} 
    stoploss = -0.05
    timeframe = '5m'
    can_short = False 

    # [실전용] 뉴스 API 호출 캐싱 변수
    last_news_call_time = datetime.min
    cached_sentiment_score = 0.0

    def feature_engineering_expand_all(self, dataframe: DataFrame, period, metadata: dict, **kwargs) -> DataFrame:
        dataframe["%rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["%mfi"] = ta.MFI(dataframe, timeperiod=14)
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe["%bb_width"] = (bollinger['upperband'] - bollinger['lowerband']) / bollinger['middleband']

        # [핵심] 오더플로우 데이터 (설정 켜져있을 때만)
        if "bid_amount" in dataframe.columns and "ask_amount" in dataframe.columns:
            dataframe["%orderflow_imbalance"] = (dataframe["bid_amount"] - dataframe["ask_amount"]) / (dataframe["bid_amount"] + dataframe["ask_amount"] + 1)
            dataframe["%cum_delta"] = dataframe["delta"].cumsum()

        # [핵심] 실행 모드에 따른 뉴스 데이터 분기 처리
        # self.dp.runmode가 존재하고, 백테스팅 모드인지 확인
        if self.dp and self.dp.runmode == RunMode.BACKTEST:
            # 1. 백테스팅: 가상 데이터(Mock) 사용
            dataframe["%sentiment_score"] = self.get_mock_sentiment_score(dataframe)
        else:
            # 2. 실전(Live/Dry): 진짜 API 호출 (캐싱 적용)
            # API 호출 비용/제한을 고려해 1시간에 한 번만 갱신
            current_score = self.fetch_real_news_sentiment()
            dataframe["%sentiment_score"] = current_score
        
        return dataframe

    def fetch_real_news_sentiment(self):
        """
        [실전용] 실제 뉴스 API를 호출하여 감성 점수를 가져옵니다.
        API 호출 횟수 제한을 피하기 위해 1시간(60분)마다 갱신합니다.
        """
        now = datetime.now()
        # 마지막 호출로부터 60분이 안 지났으면 캐시된 값 반환
        if now - self.last_news_call_time < timedelta(minutes=60):
            return self.cached_sentiment_score

        try:
            # --- [여기에 실제 API 코드를 넣으세요] ---
            # 예시: CryptoPanic API 사용 (무료 키 필요)
            # api_key = "YOUR_API_KEY"
            # url = f"https://cryptopanic.com/api/v1/posts/?auth_token={api_key}&kind=news"
            # response = requests.get(url, timeout=5)
            # data = response.json()
            
            # (임시) 실제 API가 없으므로 랜덤 값으로 대체하여 에러 방지
            # 실제 구현 시에는 여기서 텍스트 분석(NLP) 후 점수(-1~1) 리턴
            logger.info(">>> [Live] 뉴스 API 호출 및 감성 분석 수행 중...")
            
            # 예시: -0.5 ~ 0.5 사이의 값을 랜덤으로 가져온다고 가정
            real_sentiment = np.random.uniform(-0.5, 0.5) 
            
            # 캐시 업데이트
            self.cached_sentiment_score = real_sentiment
            self.last_news_call_time = now
            
            return real_sentiment

        except Exception as e:
            logger.error(f"뉴스 API 호출 실패: {e}")
            return self.cached_sentiment_score # 실패 시 이전 값 유지

    def get_mock_sentiment_score(self, dataframe):
        """
        [백테스팅용] RSI 기반 가상 감성 점수 생성 (노이즈 포함)
        """
        rsi_column = dataframe['%rsi']
        np.random.seed(42) # 결과 재현성을 위해 시드 고정
        noise = np.random.normal(0, 0.5, size=len(dataframe))
        rsi_base = (rsi_column - 50) / 50
        final_sentiment = rsi_base + noise
        return np.clip(final_sentiment, -1.0, 1.0)

    def feature_engineering_expand_basic(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe["%pct-change"] = dataframe["close"].pct_change()
        dataframe["%rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe["%rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["%-raw_close"] = dataframe["close"]
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_high"] = dataframe["high"]
        dataframe["%-raw_low"] = dataframe["low"]
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe["&-action"] = 0
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        
        # 디버깅 로그 (백테스팅 때만 출력)
        if self.dp.runmode.value == 'backtest':
            if "&-action" in dataframe.columns:
                unique_actions = dataframe['&-action'].unique()
                logger.info(f">>> [SUCCESS] {metadata['pair']} AI 행동 종류: {unique_actions}")
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        if "do_predict" in df.columns and "&-action" in df.columns:
            # 매수 조건 (3, 4, 5번 행동)
            enter_long_conditions = [
                df["do_predict"] == 1,
                (df["&-action"] >= 3) & (df["&-action"] <= 5)
            ]
            if enter_long_conditions:
                df.loc[
                    reduce(lambda x, y: x & y, enter_long_conditions),
                    ["enter_long", "enter_tag"]
                ] = (1, "long_entry")
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        if "do_predict" in df.columns and "&-action" in df.columns:
            # 매도 조건 (6, 7, 8번 행동)
            exit_long_conditions = [
                df["do_predict"] == 1,
                (df["&-action"] >= 6) & (df["&-action"] <= 8)
            ]
            if exit_long_conditions:
                df.loc[
                    reduce(lambda x, y: x & y, exit_long_conditions),
                    "exit_long"
                ] = 1
        return df