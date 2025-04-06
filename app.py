import os
from flask import Flask, render_template, jsonify
from flask_cors import CORS
import json
import threading
import time
from datetime import datetime, timedelta
import pytz
import requests
import pandas as pd
import numpy as np
from predictor import predict_next_10_candles, predict_next_prices
from analyzer import get_trading_signals
import logging
import pyupbit

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# CORS 설정 업데이트
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['TEMPLATES_AUTO_RELOAD'] = True

# 캐시 데이터
market_list = None
last_market_update = None
market_data_cache = {}
CACHE_DURATION = timedelta(minutes=1)

def get_market_list():
    global market_list, last_market_update
    
    # 캐시된 데이터가 있고 1시간이 지나지 않았으면 캐시 사용
    if market_list is not None and last_market_update is not None:
        if datetime.now() - last_market_update < timedelta(hours=1):
            return market_list
    
    try:
        # 마켓 정보 가져오기
        market_list = pyupbit.get_tickers(fiat="KRW")
        market_info = []
        
        for market in market_list:
            korean_name = market.replace("KRW-", "")  # 기본값 설정
            try:
                # 마켓 이름 정보 가져오기
                market_detail = pyupbit.get_market_info(market)
                if market_detail:
                    korean_name = market_detail.get('korean_name', korean_name)
            except Exception as e:
                print(f"Error getting market info for {market}: {e}")
            
            market_info.append({
                "market": market,
                "korean_name": korean_name
            })
        
        market_list = market_info
        last_market_update = datetime.now()
        return market_list
    except Exception as e:
        print(f"Error fetching market list: {e}")
        return []

def get_current_price(market):
    try:
        url = f"https://api.upbit.com/v1/ticker?markets={market}"
        headers = {'Accept': 'application/json'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # HTTP 오류 체크
        return response.json()[0]
    except Exception as e:
        logger.error(f"Error in get_current_price: {str(e)}")
        return None

def get_candles(market, interval="minute15", count=200):
    try:
        intervals = {
            "minute1": "minutes/1",
            "minute15": "minutes/15",
            "minute60": "minutes/60",
            "day": "days"
        }
        
        url = f"https://api.upbit.com/v1/candles/{intervals[interval]}?market={market}&count={count}"
        headers = {'Accept': 'application/json'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # HTTP 오류 체크
        data = response.json()
        
        df = pd.DataFrame(data)
        df = df.sort_values('candle_date_time_kst')
        return df
    except Exception as e:
        logger.error(f"Error in get_candles: {str(e)}")
        return pd.DataFrame()

def get_market_data(market, interval='minute15'):
    global market_data_cache
    
    cache_key = f"{market}_{interval}"
    now = datetime.now()
    
    # 캐시된 데이터가 있고 유효하면 사용
    if cache_key in market_data_cache:
        cached_data = market_data_cache[cache_key]
        if now - cached_data['timestamp'] < CACHE_DURATION:
            return cached_data['data']
    
    try:
        # 현재가 조회
        ticker = pyupbit.Ticker(market)
        current_price = ticker.trade_price
        
        # 과거 데이터 조회
        if interval == 'minute1':
            df = pyupbit.get_ohlcv(market, interval="minute1", count=60)
        elif interval == 'minute15':
            df = pyupbit.get_ohlcv(market, interval="minute15", count=48)
        elif interval == 'minute60':
            df = pyupbit.get_ohlcv(market, interval="minute60", count=24)
        elif interval == 'day':
            df = pyupbit.get_ohlcv(market, interval="day", count=30)
        else:
            df = pyupbit.get_ohlcv(market, interval="minute15", count=48)
        
        if df is None or df.empty:
            raise Exception("데이터를 가져올 수 없습니다.")
        
        # 거래량 분석
        volume_mean = df['volume'].mean()
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / volume_mean
        
        signal_type = "중립"
        confidence = 50
        
        if volume_ratio > 2:
            signal_type = "강한 매수" if df['close'].iloc[-1] > df['open'].iloc[-1] else "강한 매도"
            confidence = min(round(volume_ratio * 25), 100)
        elif volume_ratio > 1.5:
            signal_type = "매수" if df['close'].iloc[-1] > df['open'].iloc[-1] else "매도"
            confidence = min(round(volume_ratio * 20), 90)
        
        # 가격 변동 계산
        price_change = current_price - df['close'].iloc[-2]
        price_change_percent = (price_change / df['close'].iloc[-2]) * 100
        
        # 예측 데이터 (15분봉일 때만)
        predictions = None
        if interval == 'minute15':
            try:
                predictions = predict_next_prices(df)
            except Exception as e:
                print(f"Error in prediction: {e}")
                predictions = None
        
        # 캔들 데이터 준비
        candle_data = []
        for index, row in df.iterrows():
            candle_data.append({
                'candle_date_time_kst': index.strftime('%Y-%m-%d %H:%M:%S'),
                'opening_price': row['open'],
                'high_price': row['high'],
                'low_price': row['low'],
                'trade_price': row['close'],
                'candle_acc_trade_volume': row['volume']
            })
        
        # 응답 데이터 구성
        response_data = {
            'current_price': current_price,
            'price_change': price_change,
            'price_change_percent': price_change_percent,
            'signal_type': signal_type,
            'confidence': confidence,
            'candle_data': candle_data,
            'predictions': predictions
        }
        
        # 캐시 업데이트
        market_data_cache[cache_key] = {
            'timestamp': now,
            'data': response_data
        }
        
        return response_data
        
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/markets')
def api_markets():
    markets = get_market_list()
    return jsonify(markets)

@app.route('/api/market_data/<market>/<interval>')
def api_market_data(market, interval):
    data = get_market_data(market, interval)
    if data is None:
        return jsonify({'error': '데이터를 가져올 수 없습니다.'}), 500
    return jsonify(data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port) 