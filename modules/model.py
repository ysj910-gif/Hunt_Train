import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, future_steps=1, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.future_steps = future_steps
        self.num_classes = num_classes
        
        # [핵심] 레이어 설정 (Dropout 적용)
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # [핵심] 출력층: 미래의 N개 스텝을 모두 예측하기 위해 출력 노드 개수를 늘림
        # 예: 행동 클래스가 10개이고 미래 30프레임을 예측한다면 -> 출력 300개
        self.fc = nn.Linear(hidden_size, num_classes * future_steps)

    def forward(self, x):
        # 초기 Hidden State / Cell State
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM 실행
        out, _ = self.lstm(x, (h0, c0))
        
        # Many-to-One: 마지막 타임스텝의 정보만 사용하여 미래 전체를 예측
        out = self.fc(out[:, -1, :])
        
        # 출력 형태 변환: (Batch, Future_Steps, Num_Classes)
        return out.reshape(-1, self.future_steps, self.num_classes)