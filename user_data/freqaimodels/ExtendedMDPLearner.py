from typing import Any, Dict, Tuple
import numpy as np
from gymnasium import spaces
from freqtrade.freqai.RL.Base3ActionRLEnv import Base3ActionRLEnv
# [핵심] 다시 멀티프로세싱(multiproc) 모델을 상속받습니다.
from freqtrade.freqai.prediction_models.ReinforcementLearner_multiproc import ReinforcementLearner_multiproc

class ExtendedMDPLearner(ReinforcementLearner_multiproc):
    def set_freqai_targets(self, dataframe, **kwargs):
        dataframe["&s_close"] = dataframe["close"].shift(-1) / dataframe["close"] - 1
        return dataframe

class MyRLEnv(Base3ActionRLEnv):
    """
    [졸업 프로젝트 최종본] Extended MDP Environment
    - 멀티프로세싱 활성화 (속도 향상)
    """
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        # 9개 행동 (매매 3가지 x 도구 3가지)
        self.action_space = spaces.Discrete(9)

    # [Maskable PPO] 유효 행동 마스킹
    def action_masks(self) -> list[bool]:
        masks = [True] * 9
        is_long = (self._position == 1)
        is_neutral = (self._position == 0)

        if is_long:
            masks[3] = False # 매수 불가능
            masks[4] = False
            masks[5] = False
        elif is_neutral:
            masks[6] = False # 매도 불가능
            masks[7] = False
            masks[8] = False
            
        return masks

    def calculate_reward(self, action: int) -> float:
        trade_action = action // 3  
        tool_action = action % 3    
        
        # 설정 파일에서 파라미터 로드
        params = self.rl_config.get('model_reward_parameters', {})
        reward_buy = params.get('buy_reward', 1.0)
        penalty_neutral = params.get('neutral_penalty', -0.05)
        
        r_outcome = self.get_trade_pnl()
        r_process = 0.0
        
        current_candle = self.df.iloc[self.current_step]
        sentiment = current_candle['%sentiment_score']
        volatility = current_candle['%bb_width']
        
        # [PRM 로직]
        if sentiment < -0.5 and trade_action == 1:
            r_process -= 2.0 
        elif sentiment > 0.5 and trade_action == 1:
            r_process += 1.0

        if volatility > 0.05:
            if tool_action == 1: 
                r_process += 0.5
            elif tool_action == 2:
                r_process -= 0.5

        # 탐험 보상
        r_exploration = 0.0
        if trade_action == 1: 
            r_exploration = reward_buy
        elif trade_action == 0: 
            r_exploration = penalty_neutral

        return r_outcome + r_process + r_exploration

    def _perform_action(self, action: int) -> bool:
        trade_action = action // 3
        tool_action = action % 3
        
        # 도구 적용
        if tool_action == 1:
            self.stoploss = -0.02 
        elif tool_action == 2:
            self.stoploss = -0.10 
        else:
            self.stoploss = -0.05 
            
        return super()._perform_action(trade_action)