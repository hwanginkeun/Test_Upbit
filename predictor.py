import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import time
import tensorflow as tf

# 모델 캐시를 위한 전역 변수
cached_model = None
last_training_time = 0
TRAINING_CACHE_DURATION = 3600  # 1시간마다 재학습

class PricePredictor(nn.Module):
    def __init__(self, input_size=10, hidden_size=32):
        super(PricePredictor, self).__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def prepare_sequences(data, seq_length=10):
    sequences = []
    targets = []
    
    # 정규화를 위한 전체 데이터의 최대값과 최소값 계산
    data_min = data.min()
    data_max = data.max()
    
    # Min-Max 정규화 적용
    normalized_data = (data - data_min) / (data_max - data_min)
    
    for i in range(len(normalized_data) - seq_length):
        seq = normalized_data[i:i + seq_length]
        target = normalized_data[i + seq_length]
        sequences.append(seq.reshape(-1, 1))
        targets.append(target)
    
    return np.array(sequences), np.array(targets), data_min, data_max

def get_prediction(df, future_steps=5):
    global cached_model, last_training_time
    current_time = time.time()
    
    # 종가 데이터 추출
    prices = df['trade_price'].values
    
    # 시퀀스 준비
    sequences, targets, data_min, data_max = prepare_sequences(prices)
    X = torch.FloatTensor(sequences)
    y = torch.FloatTensor(targets)
    
    # 모델이 없거나 재학습이 필요한 경우에만 학습 수행
    if cached_model is None or (current_time - last_training_time) > TRAINING_CACHE_DURATION:
        # 모델 초기화 및 학습
        model = PricePredictor()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # 학습 횟수 감소 및 조기 종료 조건 추가
        epochs = 50  # 100에서 50으로 감소
        best_loss = float('inf')
        patience = 5
        no_improve = 0
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            # 조기 종료 검사
            if loss.item() < best_loss:
                best_loss = loss.item()
                no_improve = 0
                cached_model = model  # 최적의 모델 저장
            else:
                no_improve += 1
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        last_training_time = current_time
    else:
        model = cached_model
    
    # 예측
    model.eval()
    with torch.no_grad():
        # 마지막 시퀀스로 미래 예측
        last_sequence = torch.FloatTensor(sequences[-1:])
        predictions = []
        
        for _ in range(future_steps):
            pred = model(last_sequence)
            predictions.append(pred.item())
            
            # 다음 예측을 위해 시퀀스 업데이트
            last_sequence = torch.cat([
                last_sequence[:, 1:, :],
                pred.view(1, 1, 1)
            ], dim=1)
    
    # 예측값 역정규화
    predictions = np.array(predictions) * (data_max - data_min) + data_min
    return predictions.tolist()

def prepare_data(df, sequence_length=10):
    # 종가 데이터만 사용
    data = df['trade_price'].values.reshape(-1, 1)
    
    # 데이터 정규화
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(data) - sequence_length - 9):  # 10개 예측을 위해 9를 더 뺌
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length:i + sequence_length + 10])  # 다음 10개 예측
    
    return np.array(X), np.array(y), scaler

def train_model(df, sequence_length=10, epochs=100):
    # 데이터 준비
    X, y, scaler = prepare_data(df, sequence_length)
    
    # PyTorch 텐서로 변환
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    # 모델 초기화
    model = LSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 학습
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X)
        # y의 차원을 [batch_size, 10]로 변경
        y_reshaped = y.squeeze(-1)
        loss = criterion(outputs, y_reshaped)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model, scaler

def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def predict_next_prices(df, n_steps=5):
    try:
        print("Starting prediction...")  # 디버깅용 로그
        
        # 종가 데이터 추출
        prices = df['close'].values.reshape(-1, 1)
        if len(prices) < 15:  # 최소 데이터 포인트 체크
            print("Not enough data points for prediction")
            return None
        
        # 데이터 정규화
        scaler = MinMaxScaler()
        prices_scaled = scaler.fit_transform(prices)
        
        # 시퀀스 생성
        seq_length = 10
        X, y = create_sequences(prices_scaled, seq_length)
        
        if len(X) == 0:
            print("No sequences created")
            return None
        
        print(f"Created {len(X)} sequences")  # 디버깅용 로그
        
        # 모델 생성
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
            tf.keras.layers.LSTM(50, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        # 모델 컴파일
        model.compile(optimizer='adam', loss='mse')
        
        # 조기 종료 설정
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True
        )
        
        print("Training model...")  # 디버깅용 로그
        
        # 모델 학습
        model.fit(
            X.reshape(-1, seq_length, 1),
            y,
            epochs=100,
            batch_size=32,
            verbose=1,
            callbacks=[early_stopping]
        )
        
        # 다음 가격 예측
        last_sequence = prices_scaled[-seq_length:]
        predictions_scaled = []
        
        current_sequence = last_sequence.reshape(1, seq_length, 1)
        for _ in range(n_steps):
            next_pred = model.predict(current_sequence, verbose=0)
            predictions_scaled.append(next_pred[0])
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[0, -1, 0] = next_pred
        
        # 예측값 역정규화
        predictions = scaler.inverse_transform(np.array(predictions_scaled))
        predictions = predictions.flatten().tolist()
        
        print(f"Generated predictions: {predictions}")  # 디버깅용 로그
        
        return predictions
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None 