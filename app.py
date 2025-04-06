import os
from flask import Flask, render_template, jsonify
from flask_cors import CORS
import json
from datetime import datetime, timedelta
import pytz
import requests
import pandas as pd
import numpy as np
from predictor import predict_next_prices
import logging
import pyupbit

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# 캐시 데이터
market_list = None
last_market_update = None
market_data_cache = {}
CACHE_DURATION = timedelta(minutes=1)

def get_market_list():
    global market_list, last_market_update
    
    try:
        # 캐시된 데이터가 있고 1시간이 지나지 않았으면 캐시 사용
        if market_list is not None and last_market_update is not None:
            if datetime.now() - last_market_update < timedelta(hours=1):
                return market_list
        
        # 마켓 정보 가져오기
        logger.info("Fetching market list from Upbit")
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
                logger.error(f"Error getting market info for {market}: {e}")
            
            market_info.append({
                "market": market,
                "korean_name": korean_name
            })
        
        market_list = market_info
        last_market_update = datetime.now()
        logger.info(f"Successfully fetched {len(market_info)} markets")
        return market_list
        
    except Exception as e:
        logger.error(f"Error fetching market list: {e}")
        return []

def get_market_data(market, interval='minute15'):
    global market_data_cache
    
    try:
        cache_key = f"{market}_{interval}"
        now = datetime.now()
        
        # 캐시된 데이터가 있고 유효하면 사용
        if cache_key in market_data_cache:
            cached_data = market_data_cache[cache_key]
            if now - cached_data['timestamp'] < CACHE_DURATION:
                return cached_data['data']
        
        logger.info(f"Fetching market data for {market} with interval {interval}")
        
        # 현재가 조회
        current_price = pyupbit.get_current_price(market)
        if current_price is None:
            raise Exception("현재가를 가져올 수 없습니다.")
        
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
        predictions = []
        if interval == 'minute15':
            try:
                predicted_prices = predict_next_prices(df)
                if predicted_prices and len(predicted_prices) > 0:
                    predictions = predicted_prices
                    logger.info(f"Generated predictions for {market}: {predictions}")
            except Exception as e:
                logger.error(f"Error in prediction for {market}: {e}")
        
        # 캔들 데이터 준비
        candle_data = []
        for index, row in df.iterrows():
            candle_data.append({
                'candle_date_time_kst': index.strftime('%Y-%m-%d %H:%M:%S'),
                'opening_price': float(row['open']),
                'high_price': float(row['high']),
                'low_price': float(row['low']),
                'trade_price': float(row['close']),
                'candle_acc_trade_volume': float(row['volume'])
            })
        
        # 응답 데이터 구성
        response_data = {
            'current_price': float(current_price),
            'price_change': float(price_change),
            'price_change_percent': float(price_change_percent),
            'signal_type': signal_type,
            'confidence': confidence,
            'candle_data': candle_data,
            'predictions': [float(p) for p in predictions] if predictions else None
        }
        
        # 캐시 업데이트
        market_data_cache[cache_key] = {
            'timestamp': now,
            'data': response_data
        }
        
        logger.info(f"Successfully fetched market data for {market}")
        return response_data
        
    except Exception as e:
        logger.error(f"Error fetching market data for {market}: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/markets')
def api_markets():
    try:
        markets = get_market_list()
        return jsonify(markets)
    except Exception as e:
        logger.error(f"Error in /api/markets: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/market_data/<market>/<interval>')
def api_market_data(market, interval):
    try:
        data = get_market_data(market, interval)
        if data is None:
            return jsonify({'error': '데이터를 가져올 수 없습니다.'}), 500
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error in /api/market_data: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False) 