import torch, pyarrow.parquet as pq
import numpy as np

class RangeNetDataset(torch.utils.data.Dataset):
    def __init__(self, parquet_path, split="train", actions_file=None, as_distribution=True):
        self.table = pq.read_table(parquet_path).to_pandas()
        self.df = self.table[self.table["split"]==split].reset_index(drop=True)
        self.as_distribution = as_distribution
        # load actions
        if actions_file:
            with open(actions_file) as f:
                self.actions = [l.strip() for l in f if l.strip()]
        else:
            # fallback from first row
            self.actions = list(range(len(self.df.iloc[0]["y_vec"])))

        # simple categorical coders
        self.pos_vocab = {p:i for i,p in enumerate(sorted(self.df["hero_pos"].unique()))}
        self.ctx_vocab = {c:i for i,c in enumerate(sorted(self.df["ctx"].unique()))}

    def __len__(self): return len(self.df)

    def encode_cards(self, hole, board):
        # very simple char-level encoding "AsKd" + "Ah7d2c..."
        s = hole + "|" + (board or "")
        # map to ints  (A23456789TJQK|shdc|)
        vocab = "A23456789TJQKshdc|"
        idx = [vocab.index(ch) if ch in vocab else 0 for ch in s]
        arr = np.array(idx, dtype=np.int16)
        return arr

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x_num = np.array([
            float(r["stack"]),
            float(r["spr"]) if r["spr"]==r["spr"] else 0.0,  # NaN-safe
            float(r["pot"]) if r["pot"]==r["pot"] else 0.0
        ], dtype=np.float32)
        x_cat = np.array([
            self.pos_vocab[r["hero_pos"]],
            self.ctx_vocab[r["ctx"]],
        ], dtype=np.int16)
        x_cards = self.encode_cards(r["hole"], r["board"])

        if self.as_distribution:
            y = np.array(r["y_vec"], dtype=np.float32)
        else:
            y = np.int64(r["y_idx"])

        sample = {
            "x_num": torch.from_numpy(x_num),
            "x_cat": torch.from_numpy(x_cat),
            "x_cards": torch.from_numpy(x_cards),
            "y": torch.from_numpy(y) if isinstance(y, np.ndarray) else torch.tensor(y),
        }
        return sample