import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import time

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

def predict_next_10_candles(model, df, scaler, sequence_length=10):
    # 최근 데이터 준비
    recent_data = df['trade_price'].values[-sequence_length:].reshape(-1, 1)
    recent_data = scaler.transform(recent_data)
    
    # 예측
    model.eval()
    with torch.no_grad():
        X = torch.FloatTensor(recent_data).unsqueeze(0)
        predictions = model(X)
        predictions = scaler.inverse_transform(predictions.numpy())
    
    return predictions[0] 