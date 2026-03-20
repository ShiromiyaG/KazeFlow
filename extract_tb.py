"""
Extrai dados do TensorBoard e salva como CSV resumido + imprime análise.
Uso: python extract_tb.py [pasta_logs] [saida.csv]
Padrão: logs/pretrain/logs -> tb_data.csv
"""
import sys
import csv
import os
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOGS_DIR = sys.argv[1] if len(sys.argv) > 1 else "logs/pretrain/logs"
OUT_CSV  = sys.argv[2] if len(sys.argv) > 2 else "tb_data.csv"

# ── 1. Carrega todos os event files ──────────────────────────────────────────
print(f"Lendo eventos de: {LOGS_DIR}")
ea = EventAccumulator(LOGS_DIR, size_guidance={"scalars": 0})
ea.Reload()

tags = ea.Tags()["scalars"]
print(f"Métricas encontradas ({len(tags)}): {tags}\n")

# ── 2. Agrupa por step ────────────────────────────────────────────────────────
data = defaultdict(dict)   # data[step][metric] = value
for tag in tags:
    for event in ea.Scalars(tag):
        data[event.step][tag] = event.value

steps = sorted(data.keys())
print(f"Steps: {steps[0]} → {steps[-1]}  ({len(steps)} entradas)\n")

# ── 3. Salva CSV (Step, métrica1, métrica2, ...) ──────────────────────────────
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step"] + tags)
    for s in steps:
        writer.writerow([s] + [data[s].get(t, "") for t in tags])

print(f"CSV salvo em: {OUT_CSV}")

# ── 4. Análise automática ─────────────────────────────────────────────────────
def avg(iterable):
    lst = list(iterable)
    return sum(lst) / len(lst) if lst else None

def get(metric, step_min, step_max):
    return [data[s][metric] for s in steps
            if step_min <= s <= step_max and metric in data[s]]

GAN_START = next((s for s in steps if "loss/disc" in data[s]), None)
print(f"\n{'='*60}")
print(f"GAN warmup terminou no step: {GAN_START}")
print(f"{'='*60}")

# Divide em fases de ~10k steps após GAN_START
if GAN_START:
    total_gan_steps = steps[-1] - GAN_START
    phase_size = max(1000, total_gan_steps // 5)
    print(f"\n{'Fase':<20} {'cfm':>8} {'mel':>8} {'gen':>8} {'disc':>8} {'fake':>7} {'gn/flow':>9} {'gn/voc':>8}")
    print("-" * 80)
    phase = 0
    s0 = GAN_START
    while s0 < steps[-1]:
        s1 = min(s0 + phase_size, steps[-1])
        label = f"Step {s0}–{s1}"
        cfm   = avg(get("loss/cfm",    s0, s1))
        mel   = avg(get("loss/mel",    s0, s1))
        gen   = avg(get("loss/gen",    s0, s1))
        disc  = avg(get("loss/disc",   s0, s1))
        fake  = avg(get("d_score/fake",s0, s1))
        gnf   = avg(get("grad_norm/flow",   s0, s1))
        gnv   = avg(get("grad_norm/vocoder",s0, s1))
        def fmt(v): return f"{v:.3f}" if v is not None else "  N/A"
        print(f"{label:<20} {fmt(cfm):>8} {fmt(mel):>8} {fmt(gen):>8} {fmt(disc):>8} {fmt(fake):>7} {fmt(gnf):>9} {fmt(gnv):>8}")
        s0 = s1 + 1
        phase += 1

# Últimos 500 steps detalhados
print(f"\n{'='*60}")
print("Últimos 500 steps (tendência recente):")
print(f"{'='*60}")
recent = [(s, data[s]) for s in steps if s >= steps[-1] - 500]
keys_to_show = ["loss/cfm", "loss/mel", "loss/gen", "d_score/fake",
                "grad_norm/flow", "grad_norm/vocoder"]
for key in keys_to_show:
    vals = [d[key] for s, d in recent if key in d]
    if not vals:
        continue
    mid   = len(vals) // 2
    first = vals[:max(1, len(vals)//4)]
    last  = vals[max(1, 3*len(vals)//4):]
    delta = avg(last) - avg(first) if avg(first) and avg(last) else 0
    trend = "↓" if delta < -0.01 else ("↑" if delta > 0.01 else "→")
    print(f"  {key:<25} avg={avg(vals):>9.3f}  {trend}  (início={avg(first):.3f} → fim={avg(last):.3f})")

# Alerta de grad spikes
print(f"\n{'='*60}")
print("Alertas de gradiente (todo o histórico GAN):")
if GAN_START:
    for metric, clip in [("grad_norm/vocoder", 500), ("grad_norm/flow", 150)]:
        vals_over = [(s, data[s][metric]) for s in steps
                     if s >= GAN_START and metric in data[s] and data[s][metric] > clip]
        total = len([s for s in steps if s >= GAN_START and metric in data[s]])
        pct = 100 * len(vals_over) / total if total else 0
        print(f"  {metric} > {clip}: {len(vals_over)}/{total} steps = {pct:.0f}% clipados")
        if vals_over:
            worst = sorted(vals_over, key=lambda x: -x[1])[:3]
            print(f"    Top 3 piores: {[(s, round(v)) for s,v in worst]}")
