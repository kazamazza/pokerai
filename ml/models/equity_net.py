import torch, torch.nn as nn, torch.nn.functional as F

class EquityNet(nn.Module):
    def __init__(self,
                 hand_vocab=169,
                 board_vocab=256,
                 bucket_vocab=6,
                 emb_dim=32,
                 hidden=128):
        super().__init__()
        # Embeddings
        self.emb_hand   = nn.Embedding(hand_vocab, emb_dim)
        self.emb_board  = nn.Embedding(board_vocab, emb_dim)
        self.emb_bucket = nn.Embedding(bucket_vocab, emb_dim)

        # MLP: hand(E) + board(E) + bucket(E) + opp_emb(32) + board_feats(8)
        concat_dim = emb_dim*3 + 32 + 8
        self.fc1 = nn.Linear(concat_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(p=0.05)

        # init (optional but nice)
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, batch):
        # [B,E] embeddings
        e_hand  = self.emb_hand(batch["hand_id"])
        e_board = self.emb_board(batch["board_cluster_id"])

        # bucket embedding with mask (bucket_id == -1 → zeros)
        bucket_id = batch["bucket_id"]                      # [B]
        e_bucket  = self.emb_bucket(torch.clamp(bucket_id, min=0))  # [B,E]
        mask = (bucket_id >= 0).unsqueeze(-1)               # [B,1] for broadcast
        e_bucket = torch.where(mask, e_bucket, torch.zeros_like(e_bucket))

        # continuous parts come padded from the dataset to fixed sizes
        opp_cont   = batch["opp_emb"]        # [B,32]
        board_cont = batch["board_feats"]    # [B,8]

        x = torch.cat([e_hand, e_board, e_bucket, opp_cont, board_cont], dim=-1)
        x = self.dropout(F.relu(self.ln1(self.fc1(x))))
        x = self.dropout(F.relu(self.ln2(self.fc2(x))))
        return torch.sigmoid(self.out(x))    # [B,1], equity in [0,1]