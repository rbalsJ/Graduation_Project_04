# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super(Encoder, self).__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        # embedding: 입력값을 emd_dim 벡터로 변경
        self.embedding = nn.Embedding(input_dim, emb_dim)

        # embedding을 입력받아 hid_dim 크기의 hidden state, cell 출력
        self.rnn = nn.LSTM(emb_dim, hid_dim//2, n_layers // 2, dropout=dropout, bidirectional=True)

        # self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # sre: [src_len, batch_size]

        embedded = self.embedding(src)

        # initial hidden state는 zero tensor
        outputs, (hidden, cell) = self.rnn(embedded)

        # output: [src_len, batch_size, hid dim * n directions]
        # hidden: [n layers * n directions, batch_size, hid dim]
        # cell: [n layers * n directions, batch_size, hid dim]

        return hidden, cell


# decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.n_layers = n_layers

        # content vector를 입력받아 emb_dim 출력이라고 적혀있지만, 그냥 trg_vocab_size를 임베딩한 것이고,
        # hidden, cell로 주는 것이 content vector이다.
        # content vector는 encoder의 hidden 크기
        self.embedding = nn.Embedding(output_dim, emb_dim)

        # embedding을 입력받아 hid_dim 크기의 hidden state, cell 출력
        self.rnn = nn.LSTM(emb_dim, hid_dim // 2, n_layers, dropout=dropout)

        self.fc_out = nn.Linear(hid_dim // 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input: [batch_size]
        # hidden: [n layers * n directions, batch_size, hid dim]
        # cell: [n layers * n directions, batch_size, hid dim]

        input = input.unsqueeze(0)  # input: [1, batch_size], 첫번째 input은 <SOS>

        embedded = self.dropout(self.embedding(input))  # [1, batch_size, emd dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output: [seq len, batch_size, hid dim * n directions]
        # hidden: [n layers * n directions, batch size, hid dim]
        # cell: [n layers * n directions, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))  # [batch size, output dim]

        return prediction, hidden, cell


# Seq2Seq
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        #
        # # encoder와 decoder의 hid_dim이 일치하지 않는 경우 에러메세지
        # assert encoder.hid_dim == decoder.hid_dim, \
        #     'Hidden dimensions of encoder decoder must be equal'
        # # encoder와 decoder의 hid_dim이 일치하지 않는 경우 에러메세지
        # assert encoder.n_layers == decoder.n_layers, \
        #     'Encoder and decoder must have equal number of layers'

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [src len, batch size]
        # trg: [trg len, batch size]

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]  # 타겟 토큰 길이 얻기
        trg_vocab_size = self.decoder.output_dim  # 타겟 사전 크기

        # decoder의 output을 저장하기 위한 tensor
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # initial hidden state
        hidden, cell = self.encoder(src)

        # 첫 번째 입력값 <sos> 토큰
        input = trg[0, :]
        for t in range(1, trg_len):  # <eos> 제외하고 trg_len-1 만큼 반복
            output, hidden, cell = self.decoder(input, hidden, cell)

            # prediction 저장
            outputs[t] = output
            # teacher forcing을 사용할지, 말지 결정
            teacher_force = random.random() < teacher_forcing_ratio

            # 가장 높은 확률을 갖은 값 얻기
            top1 = output.argmax(1)

            # teacher forcing의 경우에 다음 lstm에 target token 입력
            input = trg[t] if teacher_force else top1

        return outputs