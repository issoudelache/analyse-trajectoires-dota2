import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.config import OUTPUT_EXP3, MIN_SUPPORT, MAX_LENGTH

d = json.load(open(OUTPUT_EXP3 / 'exp3_summary.json'))

print("=== TOP 10 motifs PrefixSpan (AP, N=3000) ===")
for i, m in enumerate(d['top10_motifs']):
    motif = m['motif']
    sup = m['support']
    pct = sup / 290 * 100
    s = ' -> '.join(str(x) for x in motif)
    print(f"  #{i+1}: [{s}]  support={sup} ({pct:.1f}%)  len={len(motif)}")

# Now get ALL multi-step motifs (len>=2), sorted by support
print("\n=== TOP 20 motifs de longueur >= 2 ===")
from dota_analytics.mining import PrefixSpan
miner = PrefixSpan(min_support=MIN_SUPPORT, max_length=MAX_LENGTH)
db = miner.load_spmf(str(OUTPUT_EXP3 / 'sequences_final.spmf'))
patterns = miner.mine(db, parallel=False)

multi = [(p, s) for p, s in patterns.items() if len(p) >= 2]
multi.sort(key=lambda x: -x[1])
for i, (p, s) in enumerate(multi[:20]):
    pct = s / 290 * 100
    motif_str = ' -> '.join(str(x) for x in p)
    print(f"  #{i+1}: [{motif_str}]  support={s} ({pct:.1f}%)  len={len(p)}")

print(f"\nTotal motifs len>=2: {len(multi)}")
print(f"Total motifs len>=3: {len([x for x in multi if len(x[0])>=3])}")

# Top-10 len>=3
print("\n=== TOP 10 motifs de longueur >= 3 ===")
tri = [(p, s) for p, s in patterns.items() if len(p) >= 3]
tri.sort(key=lambda x: -x[1])
for i, (p, s) in enumerate(tri[:10]):
    pct = s / 290 * 100
    motif_str = ' -> '.join(str(x) for x in p)
    print(f"  #{i+1}: [{motif_str}]  support={s} ({pct:.1f}%)  len={len(p)}")
