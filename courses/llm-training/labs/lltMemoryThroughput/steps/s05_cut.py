variants = {
    "baseline": {},
    "checkpointing": {"ckpt": True},
    "chunked CE x8": {"chunks": 8},
    "both": {"ckpt": True, "chunks": 8},
}
cut = {}
for name, sw in variants.items():
    peak = measure_peak(8, 1024, **sw)[0]
    tps, _ = tokens_per_sec(8, 1024, **sw)
    largest = None
    for b in (8, 12, 16, 24, 32, 48, 64, 96, 128):
        if measure_peak(b, 1024, **sw) is None:
            break
        largest = b
    cut[name] = (peak, tps, largest)
    print(f"{name:14} b=8 s=1024  predicted {predict_peak_gb(8, 1024, **sw):5.2f}  measured {peak:5.2f} GB  "
          f"{tps:7,.0f} tokens/s  largest batch at s=1024: {largest}")
