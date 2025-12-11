# Gabor energy diagram interpretation (TCGA example)

This note interprets the TCGA Gabor energy diagram supplied in the latest run (50% subsampling of embeddings, per-model clipping at each mean energy).

- **Conch** shows the lowest energy retention (mean ≈ 22.72), with its density concentrated below 25. This suggests its embeddings carry the least discriminative texture among the compared models.
- **Giga** rises to a mid-range energy (mean ≈ 28.96), indicating moderate morphology content.
- **Uni** sits just above 30 (mean ≈ 30.38), reflecting stronger texture preservation than conch/giga but below the top performers.
- **Phikon** and **virchow** cluster at the high end (means ≈ 31.11 and ≈ 31.28, respectively), with narrow, high-energy densities that indicate the richest diagnostic texture capture in this cohort.
- The legend accuracies (uni 0.6856, conch 0.6916, giga 0.7108, phikon 0.6675, virchow 0.6952) provide a secondary reference: giga tops accuracy despite slightly lower energy than phikon/virchow, showing that energy and accuracy are correlated but not identical signals.

Overall, the histogram/bottom-bar pairing separates models cleanly: phikon and virchow lead in morphology richness, uni/giga are mid-pack, and conch trails on texture content. Combining the energy ordering with the accuracy legend clarifies which models balance texture retention and downstream performance.

## How to read dense bars in the top histogram

- The histogram is density-normalised, so tall, narrow bars indicate that many tiles from that model cluster tightly within a short energy range. When a model’s density spikes near its mean, it signals consistent morphology capture across the sampled tiles (high stability), whereas flatter curves imply more variable texture quality.
- Comparing overlapping bars: if one model’s curve sits to the right and remains taller over most of the range, it both achieves higher typical energy **and** concentrates its tiles there—this is why virchow/phikon are visually separated from conch. If densities overlap but peak heights differ, focus on where the dominant mass lies; a model with a taller peak at higher energy still reflects better texture retention even if the curves cross elsewhere.
- Use the accuracy legend to contextualise the densities. A model with high, right-shifted density but modest accuracy (e.g., phikon vs. giga) suggests strong texture encoding that may not fully translate to downstream accuracy, while the reverse indicates efficiency despite lower raw energy.
