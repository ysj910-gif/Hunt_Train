import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, 
                 num_jobs=10, job_embed_dim=16, dropout=0.3):
        super(LSTMModel, self).__init__()
        
        # 1. 직업 임베딩 층 (직업 ID -> 벡터 변환)
        self.job_embedding = nn.Embedding(num_jobs, job_embed_dim)
        
        # 2. LSTM 입력 크기 = 센서 데이터 크기 + 직업 벡터 크기
        combined_input_size = input_size + job_embed_dim
        
        # 3. LSTM 레이어
        self.lstm = nn.LSTM(combined_input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        
        # 4. 분류기 (행동 예측용)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, job_idx): 
        # x: (Batch, Seq_Len, Input_Size)
        # job_idx: (Batch,)  <-- 1D 텐서여야 함
        
        # 1. 직업 ID를 임베딩 벡터로 변환
        # (Batch) -> (Batch, Embed_Dim)
        job_emb = self.job_embedding(job_idx)
        
        # 2. 임베딩을 시퀀스 길이에 맞춰 복사 (모든 프레임에 직업 정보 붙이기)
        # (Batch, Embed) -> (Batch, 1, Embed) -> (Batch, Seq, Embed)
        seq_len = x.size(1)
        job_emb = job_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 3. 센서 데이터와 직업 유전자 결합
        # 결과: (Batch, Seq, Input + Embed)
        x = torch.cat([x, job_emb], dim=2)

        # 4. LSTM 실행
        out, _ = self.lstm(x)
        
        # 5. 마지막 프레임의 결과만 사용하여 분류
        return self.fc(out[:, -1, :])