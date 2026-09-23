"""예측 단위(vocabulary)별 CPU 학습 비용 실측.
서비스 설계상 예측 단위를 어디로 잡느냐가 곧 학습 가능/불가능을 가르므로, 추정 대신 잰다."""
import time, torch, torch.nn as nn

torch.manual_seed(0)
D, NHEAD, FFN, L, MAXITEM, BATCH = 64, 2, 128, 10, 8, 128


class SmallTransformer(nn.Module):
    """방문(basket) 단위 시퀀스 -> 재방문 확률 + 다음 바스켓 multi-label."""
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab + 1, D, padding_idx=0)          # shared
        self.pos = nn.Parameter(torch.zeros(1, L, D))                  # shared
        enc = nn.TransformerEncoderLayer(D, NHEAD, FFN, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=1)            # shared
        self.head_revisit = nn.Linear(D, 1)                            # local
        self.head_basket = nn.Linear(D, vocab)                         # local

    def forward(self, items, mask):
        # items (B,L,MAXITEM) -> 상품 임베딩 평균으로 방문 벡터
        e = self.emb(items)
        v = (e * mask.unsqueeze(-1)).sum(2) / mask.sum(2, keepdim=True).clamp(min=1)
        h = self.enc(v + self.pos)[:, -1]
        return self.head_revisit(h).squeeze(-1), self.head_basket(h)


def bench(vocab, name, n_seq=20_000, steps=None):
    m = SmallTransformer(vocab)
    shared = sum(p.numel() for n, p in m.named_parameters() if not n.startswith("head_"))
    local = sum(p.numel() for n, p in m.named_parameters() if n.startswith("head_"))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    bce, bcel = nn.BCEWithLogitsLoss(), nn.BCEWithLogitsLoss()

    items = torch.randint(1, vocab + 1, (BATCH, L, MAXITEM))
    mask = (torch.rand(BATCH, L, MAXITEM) > 0.3).float()
    y_rev = (torch.rand(BATCH) > 0.5).float()
    y_bas = (torch.rand(BATCH, vocab) > 0.98).float()

    n_step = steps or max(1, n_seq // BATCH)
    for _ in range(3):                                     # warmup
        opt.zero_grad(); r, b = m(items, mask)
        (bce(r, y_rev) + bcel(b, y_bas)).backward(); opt.step()

    t0 = time.perf_counter()
    for _ in range(n_step):
        opt.zero_grad(); r, b = m(items, mask)
        (bce(r, y_rev) + bcel(b, y_bas)).backward(); opt.step()
    dt = time.perf_counter() - t0

    print(f"{name:<34} vocab={vocab:>6,}  shared={shared:>9,}  local={local:>10,}  "
          f"1 local epoch({n_seq:,} seq) = {dt:6.2f}s")
    return dt


print(f"threads={torch.get_num_threads()}  batch={BATCH}  d_model={D}  layers=1  heads={NHEAD}\n")
t_dept = bench(44,     "DEPARTMENT (44)")
t_comm = bench(308,    "COMMODITY_DESC (~308)")
t_sub  = bench(2383,   "SUB_COMMODITY_DESC (~2,383)")
t_prod = bench(92000,  "PRODUCT_ID (~92,000)")

print("\n--- FL 전체 실험 비용 환산 (4 clients x 1 local epoch/round) ---")
for name, t in [("COMMODITY (308)", t_comm), ("SUB_COMMODITY (2,383)", t_sub), ("PRODUCT_ID (92,000)", t_prod)]:
    r200 = t * 4 * 200 / 60
    print(f"{name:<24} FL 200 rounds = {r200:6.1f} min   |  "
          f"전체 실험(FL+Local+Central+재학습, ~5회) = {r200*5/60:5.2f} h")
