import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from flask import Flask, render_template, request, send_file
import io
import base64
from predictor import get_prediction
from analyzer import get_trading_signals

app = Flask(__name__)

def get_available_markets():
    """업비트에서 원화 마켓의 암호화폐 목록을 가져옴"""
    url = "https://api.upbit.com/v1/market/all"
    response = requests.get(url)
    data = response.json()
    
    # 원화 마켓만 필터링
    krw_markets = {item['korean_name']: item['market'] 
                  for item in data 
                  if item['market'].startswith('KRW-')}
    
    # 일봉 데이터로 상승률 계산
    market_changes = []
    for name, market in krw_markets.items():
        try:
            # 일봉 데이터 가져오기
            url = "https://api.upbit.com/v1/candles/days"
            params = {"market": market, "count": 1}
            response = requests.get(url, params=params)
            day_data = response.json()[0]
            
            # 상승률 계산
            change_rate = ((day_data['trade_price'] - day_data['opening_price']) 
                         / day_data['opening_price'] * 100)
            
            market_changes.append({
                'name': name,
                'market': market,
                'change_rate': change_rate,
                'current_price': day_data['trade_price']
            })
        except:
            continue
    
    # 상승률 기준으로 정렬
    market_changes.sort(key=lambda x: x['change_rate'], reverse=True)
    
    # 정렬된 딕셔너리 생성
    sorted_markets = {item['name']: item['market'] for item in market_changes}
    
    return sorted_markets, market_changes

# 사용 가능한 암호화폐 목록을 API에서 가져옴
AVAILABLE_MARKETS, MARKET_CHANGES = get_available_markets()

def get_upbit_data(market="KRW-BTC", count=200, interval="minute1"):
    # interval에 따라 적절한 URL 선택
    if interval == "minute1":
        url = "https://api.upbit.com/v1/candles/minutes/1"
    elif interval == "minute60":
        url = "https://api.upbit.com/v1/candles/minutes/60"
    else:
        raise ValueError("Unsupported interval")

    params = {
        "market": market,
        "count": count
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    df = pd.DataFrame(data)
    df['candle_date_time_kst'] = pd.to_datetime(df['candle_date_time_kst'])
    # 시간순으로 정렬 (오래된 데이터부터)
    df = df.iloc[::-1].reset_index(drop=True)
    return df

def create_candlestick_plot(df, predictions=None):
    plt.figure(figsize=(15, 7))
    
    # 실제 데이터 플로팅 (캔들스틱)
    for i in range(len(df)):
        if df['trade_price'].iloc[i] >= df['opening_price'].iloc[i]:
            color = 'red'
        else:
            color = 'blue'
            
        plt.plot([i, i], [df['low_price'].iloc[i], df['high_price'].iloc[i]], color=color, linewidth=1)
        plt.plot([i, i], [df['opening_price'].iloc[i], df['trade_price'].iloc[i]], color=color, linewidth=3)
    
    # 예측값 플로팅 (녹색으로 표시)
    if predictions is not None:
        last_index = len(df) - 1
        x_pred = range(last_index + 1, last_index + len(predictions) + 1)
        plt.plot(x_pred, predictions, 'g-', linewidth=1, label='Predicted Price')
        plt.plot(x_pred, predictions, 'go', markersize=3)  # 예측 포인트 크기를 3으로 줄임
    
    plt.title('Upbit 가격 차트')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()
    
    # 그래프를 이미지로 변환
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_data = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_data

@app.route('/', methods=['GET', 'POST'])
def index():
    # 비트코인의 한글 이름 찾기
    btc_korean_name = next((name for name, code in AVAILABLE_MARKETS.items() 
                           if code == 'KRW-BTC'), '비트코인')
    
    selected_market = request.form.get('market', btc_korean_name)
    market_code = AVAILABLE_MARKETS.get(selected_market)
    
    if market_code is None:
        # 잘못된 마켓이 선택된 경우 비트코인으로 기본 설정
        selected_market = btc_korean_name
        market_code = 'KRW-BTC'
    
    # 1분봉 데이터 가져오기
    df_1m = get_upbit_data(market=market_code, interval="minute1")
    # 1시간봉 데이터 가져오기 (예측용)
    df = get_upbit_data(market=market_code, interval="minute60")
    
    # 거래량 분석 및 매매 시그널 얻기
    trading_signals = get_trading_signals(df_1m)
    
    # 가격 예측
    predictions = get_prediction(df)
    
    # 차트 생성
    plot_data = create_candlestick_plot(df, predictions)
    
    # 현재가는 데이터프레임의 마지막 행
    current_price = df.iloc[-1]['trade_price']
    price_change = df.iloc[-1]['trade_price'] - df.iloc[-1]['opening_price']
    price_change_percent = (price_change / df.iloc[-1]['opening_price']) * 100
    
    # 10시간 후 예측값의 변동률 계산
    last_prediction = predictions[-1]
    prediction_change = last_prediction - current_price
    prediction_change_percent = (prediction_change / current_price) * 100
    
    return render_template('index.html',
                         markets=AVAILABLE_MARKETS,
                         market_changes=MARKET_CHANGES,
                         selected_market=selected_market,
                         plot_data=plot_data,
                         current_price=format(current_price, ','),
                         price_change=format(abs(price_change), ','),
                         price_change_percent=round(price_change_percent, 2),
                         is_price_up=price_change > 0,
                         predictions=[format(p, ',.0f') for p in predictions],
                         prediction_change=format(abs(prediction_change), ','),
                         prediction_change_percent=round(prediction_change_percent, 2),
                         is_prediction_up=prediction_change > 0,
                         signal_type=trading_signals['signal_type'],
                         signal_confidence=trading_signals['confidence'])

if __name__ == '__main__':
    app.run(debug=True)
