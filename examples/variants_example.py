from bushido_mythos import (
    mythos_1b,
    BushidoMythos,
)

cfg = mythos_1b()
model = BushidoMythos(cfg)

total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total:,}")
