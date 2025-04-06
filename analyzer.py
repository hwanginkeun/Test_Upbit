import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def calculate_vwap(df):
    """거래량 가중 평균 가격(VWAP) 계산"""
    df['vwap'] = (df['trade_price'] * df['candle_acc_trade_volume']).cumsum() / df['candle_acc_trade_volume'].cumsum()
    return df

def calculate_rsi(df, period=14):
    """상대강도지수(RSI) 계산"""
    delta = df['trade_price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_obv(df):
    """On-Balance Volume 계산"""
    obv = (np.sign(df['trade_price'].diff()) * df['candle_acc_trade_volume']).cumsum()
    return obv

def analyze_volume(df):
    """거래량 분석 및 매수/매도 시그널 생성"""
    # VWAP 계산
    df = calculate_vwap(df)
    
    # RSI 계산
    df['rsi'] = calculate_rsi(df)
    
    # OBV 계산
    df['obv'] = calculate_obv(df)
    
    # 거래량 증가율 계산
    df['volume_change'] = df['candle_acc_trade_volume'].pct_change()
    
    # 거래량 이동평균
    df['volume_ma5'] = df['candle_acc_trade_volume'].rolling(window=5).mean()
    df['volume_ma20'] = df['candle_acc_trade_volume'].rolling(window=20).mean()
    
    # 매수/매도 시그널 생성
    signals = pd.DataFrame(index=df.index)
    
    # 매수 조건:
    # 1. 거래량이 5분 이동평균보다 50% 이상 높음
    # 2. RSI가 30 이하 (과매도)
    # 3. 현재가가 VWAP보다 낮음
    buy_conditions = (
        (df['candle_acc_trade_volume'] > df['volume_ma5'] * 1.5) &
        (df['rsi'] < 30) &
        (df['trade_price'] < df['vwap'])
    )
    
    # 매도 조건:
    # 1. 거래량이 5분 이동평균보다 50% 이상 높음
    # 2. RSI가 70 이상 (과매수)
    # 3. 현재가가 VWAP보다 높음
    sell_conditions = (
        (df['candle_acc_trade_volume'] > df['volume_ma5'] * 1.5) &
        (df['rsi'] > 70) &
        (df['trade_price'] > df['vwap'])
    )
    
    signals['signal'] = 0  # 중립
    signals.loc[buy_conditions, 'signal'] = 1  # 매수
    signals.loc[sell_conditions, 'signal'] = -1  # 매도
    
    # 신뢰도 점수 계산 (0-100)
    confidence = pd.Series(index=df.index, dtype=float)
    
    # 매수 신뢰도
    buy_confidence = (
        (30 - df['rsi']) / 30 * 40 +  # RSI 기여도 (최대 40점)
        (df['candle_acc_trade_volume'] / df['volume_ma5'] - 1) * 30 +  # 거래량 기여도 (최대 30점)
        (df['vwap'] / df['trade_price'] - 1) * 30  # VWAP 기여도 (최대 30점)
    )
    
    # 매도 신뢰도
    sell_confidence = (
        (df['rsi'] - 70) / 30 * 40 +  # RSI 기여도 (최대 40점)
        (df['candle_acc_trade_volume'] / df['volume_ma5'] - 1) * 30 +  # 거래량 기여도 (최대 30점)
        (df['trade_price'] / df['vwap'] - 1) * 30  # VWAP 기여도 (최대 30점)
    )
    
    confidence[buy_conditions] = buy_confidence[buy_conditions].clip(0, 100)
    confidence[sell_conditions] = sell_confidence[sell_conditions].clip(0, 100)
    signals['confidence'] = confidence
    
    return signals

def get_trading_signals(df):
    """최종 매매 시그널 및 분석 결과 반환"""
    signals = analyze_volume(df)
    
    # 현재 시점의 시그널과 신뢰도 추출
    current_signal = signals['signal'].iloc[-1]
    current_confidence = signals['confidence'].iloc[-1]
    
    # 시그널 해석
    if current_signal == 1:
        signal_type = "매수"
    elif current_signal == -1:
        signal_type = "매도"
    else:
        signal_type = "관망"
    
    return {
        'signal_type': signal_type,
        'confidence': round(current_confidence, 2) if not np.isnan(current_confidence) else 0,
        'signals': signals
    } 