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

app = Flask(__name__)
CORS(app)  # CORS 지원 추가
app.config['TEMPLATES_AUTO_RELOAD'] = True

def get_market_list():
    url = "https://api.upbit.com/v1/market/all"
    response = requests.get(url)
    markets = response.json()
    krw_markets = [market for market in markets if market['market'].startswith('KRW-')]
    return krw_markets

def get_current_price(market):
    url = f"https://api.upbit.com/v1/ticker?markets={market}"
    response = requests.get(url)
    return response.json()[0]

def get_candles(market, interval="minute15", count=200):
    intervals = {
        "minute1": "minutes/1",
        "minute15": "minutes/15",
        "minute60": "minutes/60",
        "day": "days"
    }
    
    url = f"https://api.upbit.com/v1/candles/{intervals[interval]}?market={market}&count={count}"
    response = requests.get(url)
    data = response.json()
    
    df = pd.DataFrame(data)
    df = df.sort_values('candle_date_time_kst')
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/markets')
def get_markets():
    markets = get_market_list()
    return jsonify(markets)

@app.route('/api/market_data/<market>/<interval>')
def get_market_data(market, interval="minute15"):
    try:
        # 캔들 데이터 가져오기
        df = get_candles(market, interval)
        
        # 현재가 정보 가져오기
        current_price_info = get_current_price(market)
        
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
            predictions = predict_next_10_candles(df)
        
        response_data = {
            'current_price': current_price,
            'price_change': price_change,
            'price_change_percent': price_change_percent,
            'signal_type': trading_signals['signal_type'],
            'confidence': trading_signals['confidence'],
            'candle_data': df.to_dict('records'),
            'predictions': predictions.tolist() if predictions is not None else None
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port) 