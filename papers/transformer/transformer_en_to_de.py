import torch
import torch.nn as nn
import torch.optim as optim

import time
import math
import random

import spacy
import en_core_web_sm
from nltk.tokenize import word_tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

spacy_en = en_core_web_sm.load() # 영어 toknizer
spacy_de = spacy.load('de_core_news_sm') # 독일어 toknizer

'''
# 간단히 토큰화(tokenization) 기능 써보기
tokenized = spacy_en.tokenizer("I am a graduate student.")

for i, token in enumerate(tokenized):
    print(f"인덱스 {i}: {token.text}")
'''

# 영어(English) 문장을 토큰화 하는 함수 (순서를 뒤집지 않음)
def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]

# 독일어(Deutsch) 문장을 토큰화 하는 함수 
def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]


from torchtext.data import Field, BucketIterator

SRC = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", lower=True, batch_first=True)
TRG = Field(tokenize=tokenize_de, init_token="<sos>", eos_token="<eos>", lower=True, batch_first=True)
# batch가 첫번째 차원

from torchtext.datasets import Multi30k # 단어풀 쉽게 다운받아 사용 가능

train_dataset, valid_dataset, test_dataset = Multi30k.splits(exts=(".en", ".de"), fields=(SRC, TRG))
    
print(f"학습 데이터셋(training dataset) 크기: {len(train_dataset.examples)}개")
print(f"평가 데이터셋(validation dataset) 크기: {len(valid_dataset.examples)}개")
print(f"테스트 데이터셋(testing dataset) 크기: {len(test_dataset.examples)}개")

# 학습 데이터 중 하나를 선택해 출력
print(vars(train_dataset.examples[30])['src'])
print(vars(train_dataset.examples[30])['trg'])


# 최소 두번 이상 등장한 단어에 대해서만 vcab 에 추가함
SRC.build_vocab(train_dataset, min_freq=2)
TRG.build_vocab(train_dataset, min_freq=2)

print(f"len(SRC): {len(SRC.vocab)}")
print(f"len(TRG): {len(TRG.vocab)}")


# 무슨 숫자로 임베딩되는지 볼 수 있음
print(TRG.vocab.stoi["abcabc"]) # 없는 단어: 0
print(TRG.vocab.stoi[TRG.pad_token]) # 패딩(padding): 1
print(TRG.vocab.stoi["<sos>"]) # <sos>: 2
print(TRG.vocab.stoi["<eos>"]) # <eos>: 3
print(TRG.vocab.stoi["hello"])
print(TRG.vocab.stoi["world"])


BATCH_SIZE = 128

# BucketIterator : 일반적인 dataloader 기능이 있는데 이 dataloader를 만들 때 batch별로 비슷한 길이의 문장끼리 묶도록 함으로써 패딩을 최소화
# 토큰화 + 각 배치별 최대길이에 맞춰 패딩작업
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_dataset, valid_dataset, test_dataset),
    batch_size=BATCH_SIZE, shuffle=False,
    device=device)

for idx, batch in enumerate(test_iterator):
    src = batch.src
    trg = batch.trg

    print(f"첫 번째 배치 크기: {src.shape}")

    # 현재 배치에 있는 하나의 문장에 포함된 정보 출력
    for i in range(src.shape[1]):
        print(f"인덱스 {i}: {src[idx][i].item()}")

    break  # 첫번째 배치만 확인


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()

        assert hidden_dim % n_heads == 0
        # hidden_dim 이 n_heads로 나누어 떨어져야만 함. 그래야 n_head x head_dim = hidden_dim
        # n_head : 어텐션 헤드 개수
        # head_dim : 각 헤드의 디멘션
        # hidden_dim : 모든 어텐션의 디멘션

        self.hidden_dim = hidden_dim # 임베딩 차원
        self.n_heads = n_heads # 헤드(head)의 개수: 서로 다른 어텐션(attention) 컨셉의 수
        self.head_dim = hidden_dim // n_heads # 각 헤드(head)에서의 임베딩 차원

        self.fc_q = nn.Linear(hidden_dim, hidden_dim) # Query 값에 적용될 FC 레이어
        self.fc_k = nn.Linear(hidden_dim, hidden_dim) # Key 값에 적용될 FC 레이어
        self.fc_v = nn.Linear(hidden_dim, hidden_dim) # Value 값에 적용될 FC 레이어

        self.fc_o = nn.Linear(hidden_dim, hidden_dim) # 임베딩 디멘션, 원래 모양

        self.dropout = nn.Dropout(dropout_ratio)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device) # 어텐션에너지값을 소프트맥스 전 나누는 용도

    def forward(self, query, key, value, mask = None):

        batch_size = query.shape[0]

        # query: [batch_size, query_len, hidden_dim] // query_len = 단어의 개수
        # key: [batch_size, key_len, hidden_dim]
        # value: [batch_size, value_len, hidden_dim]
 
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q: [batch_size, query_len, hidden_dim]
        # K: [batch_size, key_len, hidden_dim]
        # V: [batch_size, value_len, hidden_dim]

        # hidden_dim → n_heads X head_dim 형태로 변형
        # n_heads(h)개의 서로 다른 어텐션(attention) 컨셉을 학습하도록 유도
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # permute(): 다차원 행렬전치에 사용. transpose()는 permute()의 두개만 쓰는 버전임

        # Q: [batch_size, n_heads, query_len, head_dim]
        # K: [batch_size, n_heads, key_len, head_dim]
        # V: [batch_size, n_heads, value_len, head_dim]

        # Attention Energy 계산
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy: [batch_size, n_heads, query_len, key_len]

        # 마스크(mask)를 사용하는 경우, encoder에서도 사용하는 이유는 여기서는 0이 없는 단어, 1이 패딩이라 패딩부분을 0으로 처리하기 위함임
        if mask is not None:
            energy = energy.masked_fill(mask==0, -1e10)
        # 마스크(mask) 값이 0인 부분을 -1e10으로 채우기 - softmax 이후 0%가 되도록
        # 마스크 벡터는 trg_pad_mask 에 저장시켜 사용하는데
        """ (마스크 예시)
        1 0 0 0 0
        1 1 0 0 0
        1 1 1 0 0
        1 1 1 1 0
        1 1 1 1 1
        """
        # 이 모양으로 되어있음
        
        # 어텐션(attention) 스코어 계산: 각 단어에 대한 확률 값
        attention = torch.softmax(energy, dim=-1)
        # attention: [batch_size, n_heads, query_len, key_len]   query_len = key_len

        # 여기에서 Scaled Dot-Product Attention을 계산 = attention value 값
        x = torch.matmul(self.dropout(attention), V)

        # x: [batch_size, n_heads, query_len, head_dim]

        x = x.permute(0, 2, 1, 3).contiguous()
        # contiguous(): 행렬 전치할 때 메모리 상 저장 형태를 바꾸는데
        # 자동으로 될 수도 있고 아닐 수도 있음. 자동으로 안되면 저거 붙여주면 됨

        # x: [batch_size, query_len, n_heads, head_dim] <<

        x = x.view(batch_size, -1, self.hidden_dim) # 콘캣
        # view(): 토치에서 이 함수는 다차원 행렬을 저차원 행렬로 변환해줌

        # x: [batch_size, query_len, hidden_dim] << 변경되는 부분 참고     n_heads x head_dim = hidden_dim
        # 이 모양은 처음에 넣었던 각 키, 쿼리, 밸류 모양과 동일함

        x = self.fc_o(x)
        # 원래 모양만든거 가지고 리니어 한번 통과해서 weight값 곱해준 것 - feedforward network 부분

        # x: [batch_size, query_len, hidden_dim]

        return x, attention # 한개의 어텐션자체와 어텐션스코어값(확률값)을 따로 뽑아 시각화에 사용


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hidden_dim, pf_dim, dropout_ratio):
        super().__init__()

        self.fc_1 = nn.Linear(hidden_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):

        # x: [batch_size, seq_len, hidden_dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x: [batch_size, seq_len, pf_dim]

        x = self.fc_2(x)

        # x: [batch_size, seq_len, hidden_dim]
        
        # 걍 렐루한번, 리니어한번 때려 나감. 포지션 벡터 차원을 하나 정해주고 그 벡터에 맞춰서 각 값의 자리별로 서로 다른 값을 갖도록 한 후
        # 다시 원래 모양으로 되돌려 나감. 그러면 각 자리의 값들은 위치벡터값을 간직하고 있는 채로 모양만 원래 모양으로 변경됨

        return x

class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_dim, pf_dim, dropout_ratio)
        self.dropout = nn.Dropout(dropout_ratio)

    # 하나의 임베딩이 복제되어 Query, Key, Value로 입력되는 방식
    def forward(self, src, src_mask):

        # src: [batch_size, src_len, hidden_dim]
        # src_mask: [batch_size, src_len]

        # self attention
        # 필요한 경우 마스크(mask) 행렬을 이용하여 어텐션(attention)할 단어를 조절 가능
        _src, _ = self.self_attention(src, src, src, src_mask)
        # self-attention 이므로 _src 에는 src키, src쿼리, src밸류

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        # 어텐션 전 입력값과 어텐션 후 입력값 더하고 layer nomalization
        # 이유는 어텐션 전 후 레이어들의 값의 편차가 클 수 있기 때문에 더하고 정규화 한번 하는 것 같음

        # src: [batch_size, src_len, hidden_dim]

        # position-wise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        # 마찬가지로 피드포워드 통과 전 후를 더했으니까 값의 차이가 심할까봐 레이어 노말 한번
        
        # src: [batch_size, src_len, hidden_dim]

        return src
    
    
class Encoder(nn.Module): # 앞의 EncoderLayer를 총 n개의 레이어만큼 겹침
    def __init__(self, input_dim, hidden_dim, n_layers, n_heads, pf_dim, dropout_ratio, device, max_length=100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hidden_dim) # 들어온 것에 대한 임베딩 (밀집벡터화)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim) # 전체에 대한 임베딩 (위치값 기억테이블 생성)
    

        self.layers = nn.ModuleList([EncoderLayer(hidden_dim, n_heads, pf_dim, dropout_ratio, device) for _ in range(n_layers)])
        # nn.ModuleList : nn.Sequential과 마찬가지로 안에 레이어를 쌓을 수 있음. 하지만 forward()가 없고 안에 담긴 layer 간의 연결도 없다
        # 그냥 여러개의 레이어를 넣은 클래스를 만드는 것 (토치에서의 module = keras의 layer)
        
        self.dropout = nn.Dropout(dropout_ratio)

        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

    def forward(self, src, src_mask):

        # src: [batch_size, src_len]
        # src_mask: [batch_size, src_len]

        batch_size = src.shape[0] # 문장의 개수
        src_len = src.shape[1] # 각 문장 중 단어가 제일 많은 문장의 단어 개수 (최대길이)

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        '''>>> torch.arange(5)
        tensor([ 0,  1,  2,  3,  4])
        >>> torch.arange(1, 4)
        tensor([ 1,  2,  3])
        >>> torch.arange(1, 2.5, 0.5)
        tensor([ 1.0000,  1.5000,  2.0000])'''
        # 1. arange(0, src_len): 0 부터 src_len 까지 실수범위
        # 2. unsqueeze(0): 0번째에 한차원 늘림 (벡터형태니까) -> (1, src_len)
        # 3. repeat(batch_size, 1): dim=0으로 batch_size만큼 반복, dim=1로 1만큼 반복 -> (batch_size, src_len)
        
        # pos: [batch_size, src_len]

        # 소스 문장의 임베딩과 위치 임베딩을 더한 것을 사용
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        # src: [batch_size, src_len, hidden_dim]  각 문장들이 밀집벡터형태로 배치개수만큼 묶여 있음

        # 모든 인코더 레이어를 차례대로 거치면서 순전파(forward) 수행
        for layer in self.layers:
            src = layer(src, src_mask)
        # 실질적으로 레이어 통과 진행시키는 부분
        # 모듈 리스트로 n개만큼 쌓은 인코더 레이어에 src를 하나씩 통과시키도록 선언해둠
        
        # src: [batch_size, src_len, hidden_dim]

        return src # 마지막 레이어의 출력을 반환
    
    
class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)
        self.encoder_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_dim, pf_dim, dropout_ratio)
        self.dropout = nn.Dropout(dropout_ratio)

    # 인코더의 출력 값(enc_src)을 어텐션(attention)하는 구조
    def forward(self, trg, enc_src, trg_mask, src_mask):

        # trg: [batch_size, trg_len, hidden_dim]
        # enc_src: [batch_size, src_len, hidden_dim]
        # trg_mask: [batch_size, trg_len]
        # src_mask: [batch_size, src_len]

        # self attention
        # 자기 자신에 대하여 어텐션(attention)
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        # 키, 쿼리, 밸류 전부 자기 자신 넣음

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        
        # trg: [batch_size, trg_len, hidden_dim]

        # encoder attention
        # 디코더의 쿼리(Query)를 이용해 인코더를 어텐션(attention)
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        #  자신(디코더)의 쿼리, 인코더의 키, 인코더의 밸류
        
        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # trg: [batch_size, trg_len, hidden_dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        # trg: [batch_size, trg_len, hidden_dim]
        # attention: [batch_size, n_heads, trg_len, src_len]
        
        return trg, attention
    
    
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, n_layers, n_heads, pf_dim, dropout_ratio, device, max_length=100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)

        self.layers = nn.ModuleList([DecoderLayer(hidden_dim, n_heads, pf_dim, dropout_ratio, device) for _ in range(n_layers)])

        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout_ratio)

        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        # trg: [batch_size, trg_len]
        # enc_src: [batch_size, src_len, hidden_dim]
        # trg_mask: [batch_size, trg_len]
        # src_mask: [batch_size, src_len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos: [batch_size, trg_len]

        # output embedding + positional encoding
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        # trg: [batch_size, trg_len, hidden_dim]

        for layer in self.layers:
            # 소스 마스크와 타겟 마스크 모두 사용
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # trg: [batch_size, trg_len, hidden_dim]
        # attention: [batch_size, n_heads, trg_len, src_len]

        # output = self.fc_out(trg)
        output = torch.log_softmax(self.fc_out(trg), dim=-1)    # 그냥 소프트맥스하면 값 a 만 나오고 성능 개구리됨
        # print(sum(output[0,0,:]))
        
        # output: [batch_size, trg_len, output_dim]

        return output, attention

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    # 소스 문장(인코더의 문장)의 <pad> 토큰에 대하여 마스크(mask) 값을 0으로 설정
    def make_src_mask(self, src):

        # src: [batch_size, src_len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask: [batch_size, 1, 1, src_len]

        return src_mask

    # 타겟 문장(디코더의 문장)에서 각 단어는 다음 단어가 무엇인지 알 수 없도록(이전 단어만 보도록) 만들기 위해 마스크를 사용
    def make_trg_mask(self, trg):

        # trg: [batch_size, trg_len]

        """ (마스크 예시)
        1 0 0 0 0
        1 1 0 0 0
        1 1 1 0 0
        1 1 1 0 0
        1 1 1 0 0
        """
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # 패딩부분 마스크
        
        # trg_pad_mask: [batch_size, 1, 1, trg_len]

        trg_len = trg.shape[1]

        """ (마스크 예시)
        1 0 0 0 0
        1 1 0 0 0
        1 1 1 0 0
        1 1 1 1 0
        1 1 1 1 1
        """
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        # 치팅방지 마스크
        # torch.tril(텐서, diagonal=정수)
        ''' 3x3 텐서일때
        diagonal=-1
        tensor([[0.0000, 0.0000, 0.0000],
        [1.1507, 0.0000, 0.0000],
        [0.3187, 1.1888, 0.0000]])
        .bool() 하면 0자리 False로 채움
        tensor([[False, False, False],
        [ True, False, False],
        [ True,  True, False]])
        
        diagonal=0  
        tensor([[-0.6865,  0.0000,  0.0000],
        [ 0.2829, -0.7228,  0.0000],
        [-0.0886, -0.6464, -0.8440]])
        
        diagonal=1
        tensor([[ 1.6846,  0.2727,  0.0000],
        [-0.2204,  1.1502, -0.6502],
        [-1.3327, -0.9748, -0.9970]])
        '''
        
        
        # trg_sub_mask: [trg_len, trg_len]

        trg_mask = trg_pad_mask & trg_sub_mask
        # 패딩값마스크와 치팅방지마스크를 & 연산자로 붙여서 디코더용 마스크 하나 생성
        # & 연산자이기 땜에 두 마스크에서 수치가 다 있는 곳만 값을 구하도록 함
        # 즉 한쪽 마스크만 적용되는 부분도 다 0으로 처리가 가능
        # & 연산자: 두 값이 같으면 그 값을 뱉고 두 값이 다르면 0 뱉음

        # trg_mask: [batch_size, 1, trg_len, trg_len]

        return trg_mask

    def forward(self, src, trg):

        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask: [batch_size, 1, 1, src_len]
        # trg_mask: [batch_size, 1, trg_len, trg_len]

        enc_src = self.encoder(src, src_mask)

        # enc_src: [batch_size, src_len, hidden_dim]
        
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output: [batch_size, trg_len, output_dim]
        # attention: [batch_size, n_heads, trg_len, src_len]
        
        return output, attention


######## Training ########
INPUT_DIM = len(SRC.vocab)  # 독일어 단어사전 개수
OUTPUT_DIM = len(TRG.vocab) # 영어 단어사전 개수
HIDDEN_DIM = 256    # 전체 디멘션
ENC_LAYERS = 4      # 인코더 레이어 개수
DEC_LAYERS = 4      # 디코더 레이어 개수
ENC_HEADS = 8       # 헤드 개수
DEC_HEADS = 8
ENC_PF_DIM = 512   
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
# 패딩 토큰 무엇인지 저장 (1 임)

# 인코더(encoder)와 디코더(decoder) 객체 선언
enc = Encoder(INPUT_DIM, HIDDEN_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
dec = Decoder(OUTPUT_DIM, HIDDEN_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

# Transformer 객체 선언
model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# 기울기가 필요한 모든 파라미터들을 더해서 표시함. 계산 가능한 파라미터 총 개수 구한 것
print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        # hastter : 속성이름을 파라미터로 주었을 때, 객체에 속성이 존재할 경우 True, 아닐 경우 False를 반환한다.
        # m(모델)에서 'weight'에 해당하는 속성이 존재하면 True반환 & 가중치 차원이 1보다 크면 가중치 초기화 진행
        nn.init.xavier_uniform_(m.weight.data)
# xavier uniform 초기화는 인풋과 아웃풋 개수를 반영해서 처음 가중치를 초기화 시킨다
# 노드개수가 너무 많으면 exploding, 적으면 gradient vanishing 문제가 생기기 때문에 앞뒤 노드 개수를 반영해 가중치를 초기화한다

model.apply(initialize_weights)


# Adam optimizer로 학습 최적화
LEARNING_RATE = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 패딩(padding)에 대해서는 값 무시
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)


# 모델 학습(train) 함수
def train(model, iterator, optimizer, criterion, clip):
    model.train() # 학습 모드
    epoch_loss = 0

    # 전체 학습 데이터를 확인하며
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        # 출력 단어의 마지막 인덱스(<eos>)는 제외
        # 입력을 할 때는 <sos>부터 시작하도록 처리
        output, _ = model(src, trg[:,:-1])
        
        # output: [배치 크기, trg_len - 1, output_dim]
        # trg: [배치 크기, trg_len]
        # torch.Size([128, 27, 5920])
        # torch.Size([128, 28])

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        # 출력 단어의 인덱스 0(<sos>)은 제외
        trg = trg[:,1:].contiguous().view(-1)

        # output: [배치 크기 * trg_len - 1, output_dim]
        # trg: [배치 크기 * trg len - 1]

        # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
        loss = criterion(output, trg)
        loss.backward() # 기울기(gradient) 계산

        # 기울기(gradient) clipping 진행
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # clip_grad_norm(): 일정 기울기 threshold를 넘기면 잘라서 너무 큰 기울기가 나오지 않도록 조절
        # threshold는 기울기가 가질 수 있는 최대 L2norm값. 사용자가 정해야함. 기울기/L2norm

        # 파라미터 업데이트
        optimizer.step()

        # 전체 손실 값 계산
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# 모델 평가(evaluate) 함수
def evaluate(model, iterator, criterion):
    model.eval() # 평가 모드
    epoch_loss = 0

    with torch.no_grad():
        # 전체 평가 데이터를 확인하며
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            # 출력 단어의 마지막 인덱스(<eos>)는 제외
            # 입력을 할 때는 <sos>부터 시작하도록 처리
            output, _ = model(src, trg[:,:-1])
            
            # output: [배치 크기, trg_len - 1, output_dim=vocab_size]
            # trg: [배치 크기, trg_len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            # 출력 단어의 인덱스 0(<sos>)은 제외
            trg = trg[:,1:].contiguous().view(-1)

            # output: [배치 크기 * trg_len - 1, output_dim]
            # trg: [배치 크기 * trg len - 1]

            # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
            loss = criterion(output, trg)

            # 전체 손실 값 계산
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# '''
N_EPOCHS = 10
CLIP = 1
best_valid_loss = float('inf') # 양의 무한대부터 시작

for epoch in range(N_EPOCHS):
    start_time = time.time() # 시작 시간 기록

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time() # 종료 시간 기록
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'transformer_german_to_english.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
    print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}')
    
    
# 학습된 모델 저장
torch.save(model.state_dict(), 'transformer_german_to_english.pt')

model.load_state_dict(torch.load('transformer_german_to_english.pt'))

test_loss = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}')
# PPL : 언어모델의 평가 방법. Perplexity. 직역 시 당혹스러운 정도. 낮을 수록 좋다
# 이전 단어로 다음 단어를 예측할 때 몇 개의 단어 후보를 고려하는지
# 단, 테스트 데이터가 충분히 많고, 언어 모델을 활용할 도메인에 적합한 테스트 데이터셋으로 구성된 경우에 한함
# 이전 단어들을 기반으로 다음 단어를 예측할 때마다 평균적으로 몇개의 단어 후보 중 정답을 찾는지, 그 수가 적을 수록 좋은 것
# 따라서 같은 테스트 데이터셋에서 언어 모델 간의 PPL 값을 비교하면 어떤 언어 모델이 우수한 성능을 보이는지 알 수 있음
# 크로스엔트로피를 지수화하면 perplexity가 됨
#=============================================================================================================================
# '''








#==================================================== 평가 및 예측 ===========================================================

# 학습된 모델 불러오기
model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
model.load_state_dict(torch.load('transformer_german_to_english.pt'))

test_loss = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}')

# 번역(translation) 함수
def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50, logging=True):
    model.eval() # 평가 모드

    if isinstance(sentence, str): # isinstance : sentence 의 자료형이 str 인지 확인. bool 반환
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # 처음에 <sos> 토큰, 마지막에 <eos> 토큰 붙이기
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]    # 처음에 Filed 객체로 선언해둠
    if logging:
        print(f"전체 소스 토큰: {tokens}")

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]     # 소스문장을 번호로 바꿈
    if logging:
        print(f"소스 문장 인덱스: {src_indexes}")

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)  # 토치텐서에 실어주기

    # 소스 문장에 따른 마스크 생성
    src_mask = model.make_src_mask(src_tensor)      # 패딩부분 가리는 용도 마스크 생성

    # 인코더(endocer)에 소스 문장을 넣어 출력 값 구하기
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    
    # 처음에는 <sos> 토큰 하나만 가지고 있도록 하기
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len): # max_len 만큼 반복해서 단어 뽑기
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)  # 1배치니까

        # 출력 문장에 따른 마스크 생성
        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        # 출력 문장에서 가장 마지막 단어만 사용
        pred_token = output.argmax(2)[:,-1].item()
        trg_indexes.append(pred_token) # 출력 문장에 더하기

        # <eos>를 만나는 순간 끝
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:     # 현재 predict한 것이 <eos> 라면 끝냄
            break

    # 각 출력 단어 인덱스를 실제 단어로 변환
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    # 첫 번째 <sos>는 제외하고 출력 문장 반환
    return trg_tokens[1:], attention


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):

    assert n_rows * n_cols == n_heads

    # 출력할 그림 크기 조절
    fig = plt.figure(figsize=(15, 25))

    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        # 어텐션(Attention) 스코어 확률 값을 이용해 그리기
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'], rotation=45)
        ax.set_yticklabels([''] + translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    
example_idx = 10
# src = vars(test_dataset.examples[example_idx])['src']
# src = tokenize_de('We became four people.')    # 번역할 문장
# src = tokenize_de('There are four people in my team.')
src = tokenize_de('two dogs are playing in the snow.')

# trg = vars(test_dataset.examples[example_idx])['trg']

print(f'소스 문장: {src}')
# print(f'타겟 문장: {trg}')

translation, attention = translate_sentence(src, SRC, TRG, model, device, logging=True)

print("모델 출력 결과:", " ".join(translation))

display_attention(src, translation, attention)

# '''
# inference 및 bleu 스코어
from torchtext.data.metrics import bleu_score

def show_bleu(data, src_field, trg_field, model, device, max_len=50):
    trgs = []
    pred_trgs = []
    index = 0

    for datum in data:
        src = vars(datum)['src']
        trg = vars(datum)['trg']

        pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len, logging=False)

        # 마지막 <eos> 토큰 제거
        pred_trg = pred_trg[:-1]

        pred_trgs.append(pred_trg)
        trgs.append([trg])

        index += 1
        if (index + 1) % 100 == 0:
            print(f"[{index + 1}/{len(data)}]")
            print(f"예측: {pred_trg}")
            print(f"정답: {trg}")

    bleu = bleu_score(pred_trgs, trgs, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])
    print(f'Total BLEU Score = {bleu*100:.2f}')

    individual_bleu1_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1, 0, 0, 0])
    individual_bleu2_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[0, 1, 0, 0])
    individual_bleu3_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[0, 0, 1, 0])
    individual_bleu4_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[0, 0, 0, 1])

    print(f'Individual BLEU1 score = {individual_bleu1_score*100:.2f}') 
    print(f'Individual BLEU2 score = {individual_bleu2_score*100:.2f}') 
    print(f'Individual BLEU3 score = {individual_bleu3_score*100:.2f}') 
    print(f'Individual BLEU4 score = {individual_bleu4_score*100:.2f}') 

    cumulative_bleu1_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1, 0, 0, 0])
    cumulative_bleu2_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1/2, 1/2, 0, 0])
    cumulative_bleu3_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1/3, 1/3, 1/3, 0])
    cumulative_bleu4_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1/4, 1/4, 1/4, 1/4])

    print(f'Cumulative BLEU1 score = {cumulative_bleu1_score*100:.2f}') 
    print(f'Cumulative BLEU2 score = {cumulative_bleu2_score*100:.2f}') 
    print(f'Cumulative BLEU3 score = {cumulative_bleu3_score*100:.2f}') 
    print(f'Cumulative BLEU4 score = {cumulative_bleu4_score*100:.2f}') 
    
show_bleu(test_dataset, SRC, TRG, model, device)
# dataset은 아직 패딩처리(버킷이터레이터)하기 전. <sos>, <eos> 안붙어있음
# '''


# Total BLEU Score = 35.21  log_softmax 결과적으로 가장 좋음
# Individual BLEU1 score = 67.90
# Individual BLEU2 score = 43.13
# Individual BLEU3 score = 27.96
# Individual BLEU4 score = 18.76
# Cumulative BLEU1 score = 67.90
# Cumulative BLEU2 score = 54.12
# Cumulative BLEU3 score = 43.43
# Cumulative BLEU4 score = 35.21

# Total BLEU Score = 34.97  without softmax
# Individual BLEU1 score = 67.59
# Individual BLEU2 score = 42.84
# Individual BLEU3 score = 27.76
# Individual BLEU4 score = 18.61
# Cumulative BLEU1 score = 67.59
# Cumulative BLEU2 score = 53.81
# Cumulative BLEU3 score = 43.16
# Cumulative BLEU4 score = 34.97

# Total BLEU Score = 0.00   with softmax
# Individual BLEU1 score = 0.00
# Individual BLEU2 score = 0.00
# Individual BLEU3 score = 0.00
# Individual BLEU4 score = 0.00
# Cumulative BLEU1 score = 0.00
# Cumulative BLEU2 score = 0.00
# Cumulative BLEU3 score = 0.00
# Cumulative BLEU4 score = 0.00