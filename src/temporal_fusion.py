def fuse_maps(maps,use_ewma=False,ewma_alpha=0.6):
    n_frames = max(len(maps), 1)
    counts = {}

    if use_ewma:
        fused = {}
        decay = 1.0 - ewma_alpha
        for m in maps:
            for key, value in m.items():
                if key in fused:
                    fused[key] = ewma_alpha * value + decay * fused[key]
                    counts[key] += 1
                else:
                    fused[key] = float(value)
                    counts[key] = 1
    else:
        fused_sums = {}
        for m in maps:
            for key, value in m.items():
                fused_sums[key] = fused_sums.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1
        fused = {key: (total / counts[key]) for key, total in fused_sums.items()}

    stability = {key: (count / n_frames) for key, count in counts.items()}
    return fused, stability
