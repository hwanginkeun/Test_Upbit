import os
from flask import Flask, render_template, jsonify
from flask_cors import CORS
import json
import threading
import time
from datetime import datetime
import pytz
import requests
import pandas as pd
import numpy as np
from predictor import predict_next_10_candles
from analyzer import get_trading_signals
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# CORS 설정 업데이트
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['TEMPLATES_AUTO_RELOAD'] = True

def get_market_list():
    try:
        url = "https://api.upbit.com/v1/market/all"
        headers = {'Accept': 'application/json'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # HTTP 오류 체크
        markets = response.json()
        krw_markets = [market for market in markets if market['market'].startswith('KRW-')]
        return krw_markets
    except Exception as e:
        logger.error(f"Error in get_market_list: {str(e)}")
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/markets')
def get_markets():
    try:
        markets = get_market_list()
        return jsonify(markets)
    except Exception as e:
        logger.error(f"Error in get_markets endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/market_data/<market>/<interval>')
def get_market_data(market, interval="minute15"):
    try:
        logger.info(f"Fetching market data for {market} with interval {interval}")
        
        # 캔들 데이터 가져오기
        df = get_candles(market, interval)
        if df.empty:
            return jsonify({'error': 'Failed to fetch candle data'}), 500
        
        # 현재가 정보 가져오기
        current_price_info = get_current_price(market)
        if not current_price_info:
            return jsonify({'error': 'Failed to fetch current price'}), 500
        
        # 가격 변화 계산
        current_price = current_price_info['trade_price']
        prev_price = current_price_info['prev_closing_price']
        
        price_change = current_price - prev_price
        price_change_percent = (price_change / prev_price) * 100
        
        # 거래량 분석 및 매매 시그널 생성
        trading_signals = get_trading_signals(df)
        
        # 다음 가격 예측
        predictions = None
        if interval == "minute15":
            try:
                predictions = predict_next_10_candles(df)
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                predictions = None
        
        response_data = {
            'current_price': current_price,
            'price_change': price_change,
            'price_change_percent': price_change_percent,
            'signal_type': trading_signals['signal_type'],
            'confidence': trading_signals['confidence'],
            'candle_data': df.to_dict('records'),
            'predictions': predictions.tolist() if predictions is not None else None
        }
        
        logger.info(f"Successfully processed market data for {market}")
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error in get_market_data endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port) 