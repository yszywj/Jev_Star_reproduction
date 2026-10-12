import json
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(r"C:\Users\ry\game_skill\micro\jev_runs")

runs = {
    "JEV-only": root / "smoke-3m-jev" / "summary.json",
    "Astra + JEV": root / "smoke-3m-astra-jev" / "summary.json",
}

labels = []
enemy_kills = []
allies_alive = []
steps_list = []

for label, path in runs.items():
    data = json.loads(path.read_text(encoding="utf-8"))
    ep = data["episodes"][0]

    labels.append(label)

    initial_enemies = ep["initial"]["enemies_alive"]
    final_enemies = ep["final"]["enemies_alive"]
    kills = initial_enemies - final_enemies

    enemy_kills.append(kills)
    allies_alive.append(ep["final"]["allies_alive"])
    steps_list.append(ep["steps"])

x = list(range(len(labels)))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar([i - width/2 for i in x], enemy_kills, width, label="Enemy kills")
ax.bar([i + width/2 for i in x], allies_alive, width, label="Allies alive")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Count")
ax.set_title("Micro Smoke Test on 3m")
ax.legend()
ax.grid(axis="y", alpha=0.25)

for i, v in enumerate(enemy_kills):
    ax.text(i - width/2, v + 0.03, str(v), ha="center")
for i, v in enumerate(allies_alive):
    ax.text(i + width/2, v + 0.03, str(v), ha="center")

fig.tight_layout()

output = root / "micro_smoke_3m.png"
fig.savefig(output, dpi=180)

print(f"Saved chart: {output}")