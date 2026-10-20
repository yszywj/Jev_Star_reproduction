# SMAC-Hard micro / SMAC-Hard 微操

[English](#english) | [简体中文](#简体中文)

## English

The current interface is `p0-v1`: schema 6 for JEV alone and schema 7 for Astra + JEV. With planning enabled, Astra generates one plan per map, reused across three episodes. JEV answers one Choice question for each living unit. Small squads share a single request; larger squads are split into batches according to the budget. All batches use the same observation while the game is paused.

Defaults are `realtime=False`, `step_mul=8`, seed 1, and the base script opponent. The game clock does not advance while waiting for the model. The environment supplies no-op actions for dead units; JEV selects every living unit's action. The earlier real-time probe remains available separately through `micro-realtime` and is not the current baseline.

| Action | Meaning |
| --- | --- |
| 0 | No-op for dead units; not offered to living units |
| 1 | Stop; cancel the current order |
| 2 / 3 / 4 / 5 | Move north / south / east / west by 2 world units |
| 6+j | Attack visible enemy e{j}; for a Medivac, heal ally u{j} instead |

SMAC's center-distance limit of 6 is retained for attack candidates, while native weapon range is described separately. Abilities, transformations, arbitrary coordinates, and target reservations are not part of the action space.

P0 includes four changes: public map background, explicit range and legal-candidate semantics, complete core teammate fields across batches, and feedback from SC2's native ResponseAction and observation errors in the next decision. An accepted command does not imply that damage has already been dealt.

| File | Responsibility |
| --- | --- |
| [policy.py](jev_micro/policy.py) | Structured state, unit questions, batching, and action decoding |
| [planner.py](jev_micro/planner.py) | Per-map Astra plans and reuse |
| [scenario.py](jev_micro/scenario.py) | Public map priors and static terrain |
| [feedback.py](jev_micro/feedback.py) | Mapping native SC2 command results and errors |
| [run.py](jev_micro/run.py) | Complete episodes, logging, clock checks, and replays |
| [suite.py](jev_micro/suite.py) | Sweeps across all maps, failure preservation, and resumption |
| [catalog.py](jev_micro/catalog.py) | Catalog of 35 maps and labels for the two development scenarios |

Run three episodes on every map from the repository root:

```powershell
py -3.10 jev_star.py micro-suite --all-maps --policies jev --episodes 3 --seed 1 --step-mul 8 --planner codex --planner-model gpt-6-astra --planner-effort medium --planner-timeout 180 --request-timeout 15 --batch-concurrency 4 --api-retries 2 --output-dir jev_runs/my-sweep
```

To resume the same configuration, add `--resume` and use the same output directory. Maps with all three episodes completed are reused; failed attempts are preserved, and retries receive new directories. The summary's `summary_file` identifies the selected valid result. Create a new experiment directory when changing the interface, plan, or parameters.

### FocusFireBudget skill

`--skill-mode` selects how the parameterized focus-fire skill participates (default `off`, which keeps the original per-unit JEV / Nearest flow byte-for-byte):

| Mode | Behavior |
| --- | --- |
| `off` | Original per-unit flow; no skill code runs. |
| `always` | The local FocusFireBudget allocator acts every step without any JEV or Astra calls — free validation of the algorithm. |
| `jev` | Each decision window, JEV answers one `team_skill` Choice — `0` = direct_control, `1` = focus_fire_budget with automatic target priority, `2` = cover_retreat, `3` = heal_assign (offered only when the snapshot has a healer AND legal heal demand), `4`… = focus_fire_budget with an explicit priority target (`4`=e0, `5`=e1, …; dynamically bound to currently visible enemies, capped at four explicit candidates). The mode is held for `--skill-max-steps` steps (default 4). `jev` mode requires `--policy jev`. |

The allocator ranks visible targets deterministically, assigns just enough attackers to cover each target's remaining health plus shields (minus still-pending predicted damage), and moves the remaining attackers to the next target. Damage is predicted from public combat profiles only; unknown weapon data, cooldowns and out-of-range attackers are assigned but never counted toward the budget, so the budget under-counts instead of inventing numbers. Dead units get action `0`; every emitted action is validated against `get_avail_agent_actions()`. Medivacs are never assigned attacks; they keep deterministic heal/move fallbacks. Skill telemetry (`skill_selected` / `skill_started` / `skill_step` / `skill_reassigned` / `skill_terminated` events, per-episode and summary `skill` stats: activations, steps, mean duration, termination reasons, reassignments, predicted damage and estimated overkill) is written to `events.jsonl`, `run.json` and `summary.json`. `micro-suite` forwards `--skill-mode` / `--skill-max-steps` and rejects `--resume` attempts that change them.

```powershell
# A: original per-unit JEV (baseline)
py -3.10 jev_star.py micro --map 3m --policy jev --planner none --skill-mode off --episodes 3
# B: always use the local skill, no model calls
py -3.10 jev_star.py micro --map 3m --policy nearest --planner none --skill-mode always --episodes 3
# C: JEV chooses between per-unit control and the skill
py -3.10 jev_star.py micro --map 3m --policy jev --planner none --skill-mode jev --skill-max-steps 4 --episodes 3
```

Limitations (first version): damage prediction assumes one imminent attack per ready, in-range attacker and does not simulate projectiles, splash or heal-over-time; the skill terminates and returns control rather than guessing when target data becomes unavailable.

### HealAssign skill

`--skill-name heal_assign` (with `--skill-mode always`) or JEV choice `3` (with `--skill-mode jev`, offered only when the snapshot actually contains Medivacs) activates HealAssign. Wounded-target demand is health-only (`max_health - health`; shield loss never makes a target healable). Each healer/target edge is scored with an explainable urgency (missing health, low ratio, recent observed loss, visible threats) minus distance cost plus a `--heal-switch-margin` (0.15) bonus for keeping the previous target, then a deterministic maximum-weight **one-to-one** matching assigns healers — several Medivacs never dogpile one target, and the classic "flexible healer must yield the shared target to the restricted healer" case is solved exactly (SciPy `linear_sum_assignment` for larger rosters, exhaustive search for small ones). Already-active heal orders are detected and re-issued so the executor's `existing_heal_order_retained` path avoids duplicate commands. An endangered healer self-preserves via the safest legal move away from visible enemies (never by drifting toward a wounded ally) and releases its reservation; a healer with no in-range legal heal edge instead approaches the most urgent out-of-range wounded ally along legal moves (approach is movement, never counted as provided healing). HealAssign controls **healer slots only** — non-healers keep the declared base policy (deterministic nearest in free `always` mode, per-unit JEV in `jev` mode, merged with a logged `skill_merge` event so every unit has exactly one action source). Events: `heal_skill_selected/started/step`, `heal_assignment_created/changed`, `heal_action_issued`, `heal_assignment_released`, `heal_skill_terminated`; summary stats add assignments created, target switches, self-preserves and no-target steps. Planned healing is never counted as restored health.

```powershell
# Free local HealAssign validation on an MMM-style map with Medivacs, zero model calls
py -3.10 jev_star.py micro --map MMM --policy nearest --planner none --skill-mode always --skill-name heal_assign --episodes 1 --output-dir jev_runs\heal-assign-local
# JEV chooses between direct control, focus fire, cover retreat and heal assign
py -3.10 jev_star.py micro --map MMM --policy jev --planner none --skill-mode jev --skill-max-steps 4 --episodes 3 --output-dir jev_runs\heal-assign-jev
```

### AdaptiveFocusFire skill

`--skill-name adaptive_focus_fire` (with `--skill-mode always`) or the appended JEV posture candidates (`adaptive_focus_fire (aggressive|balanced|defensive)`) activate the composed skill. Every step, ALL fighting units — including when nobody retreats — go through FocusFireBudget on the attacker subset (never per-unit nearest fallback); only units with clear retreat value briefly retreat via CoverRetreat's detection and move scoring (max `--adaptive-max-retreat-steps`, default 2, per unit, tracked in per-unit history that survives 4-step skill windows). Kill-capable and weapon-engaged units keep attacking; cooled-down endangered units with a safe move may dodge and then re-engage. When contact has been made and there has been no observable enemy damage for `--adaptive-stall-steps` (default 6) steps combined with a real disadvantage (low team survival without heal capacity, or a retreat loop), the controller enters a sticky All-in: survival retreats stop, everyone rejoins the fire plan, and legal approach moves remain available. Postures only retune explainable thresholds (critical ratio, recent-loss bar) in one shared template. Events: `adaptive_skill_started/step`, `adaptive_roles_updated`, `adaptive_retreat_started/ended`, `adaptive_reengaged`, `adaptive_all_in_started`, `adaptive_skill_terminated`; summary adds engage/retreat/All-in step counts, retreat unit steps, re-engagements and All-in triggers. Predicted damage remains prediction, never actual damage.

```powershell
# Free 8m comparison against test-8m-focus / test-8m-cover
py -3.10 jev_star.py micro --map 8m --policy nearest --planner none --skill-mode always --skill-name adaptive_focus_fire --adaptive-posture balanced --episodes 1 --seed 1 --step-mul 8 --output-dir jev_runs\test-8m-adaptive
```

Sharpy-inspired upgrades (all default OFF for backward compatibility; enable per ablation): `--adaptive-window-fire` counts weapons whose cooldown expires within the control window (`step_mul` + `window_tolerance`, see `skills/combat_timing.py`) as budgeted fire in FocusFireBudget and AdaptiveFocusFire's retreat/kill timing; `--adaptive-range-preserving-move` replaces CoverRetreat/Adaptive retreat scoring with range-preserving movement (`skills/movement_scoring.py`, TacticalReposition vs EmergencyRetreat, melee approach bias, anti ping-pong via per-unit move history); `--adaptive-target-scoring` adds bounded target-value terms (damage taken, proximity, overkill penalty) after the kill-cost constraint. Predicted damage remains split into `immediate_damage` / `window_only_damage` diagnostics and is never actual observed damage.

### CoverRetreat skill

`--skill-name cover_retreat` (with `--skill-mode always`) or JEV choice `2` (with `--skill-mode jev`) activates CoverRetreat. Endangered allies — visible threat present AND effective survival ratio below `--cover-risk-threshold` (0.35) or a large recent loss — retreat along the best-scoring legal move (max distance from visible enemies, avoid other enemies' danger radius, prefer allied support, keep the previous direction); remaining allies with legal attacks cover them by delegating to FocusFireBudget restricted to cover-role units, so retreating units never contribute to the same step's damage budget. If everyone is endangered the whole team retreats; no unit is sacrificed as a forced cover. Exit uses hysteresis: the skill ends after `--cover-safe-streak` (2) consecutive threat-free steps rather than the moment health looks low, preventing retreat/attack oscillation. Thresholds live in one `PARAMS` dict in `skills/cover_retreat.py` and are CLI-configurable. Cover attacks are never assumed to redirect enemy targeting; Medivacs are never assigned attacks but do flee when threatened. Events: `cover_retreat_selected/started/step`, `cover_roles_updated`, `retreat_direction_selected`, `cover_retreat_terminated`; per-episode and summary `skill` stats add role reassignments, retreat unit steps and safe-streak exits.

```powershell
# Free local CoverRetreat validation, zero model calls
py -3.10 jev_star.py micro --map 3m --policy nearest --planner none --skill-mode always --skill-name cover_retreat --episodes 3
# JEV chooses between direct control, focus fire and cover retreat
py -3.10 jev_star.py micro --map 3m --policy jev --planner none --skill-mode jev --skill-max-steps 4 --episodes 3
```


`SMAC_HARD_maps/` contains all 35 registered maps, including maps from the older upstream directories. The environment uses the modified `pysc2/` bundled here; do not replace it with a separate PySC2 package installed through pip.

See [experiments](../docs/experiments.md) for results and the [main README](../README.md#english) for installation and API key configuration.

## 简体中文

当前接口为 `p0-v1`：纯 JEV 为 schema 6，Astra＋JEV 为 schema 7。每图先由 Astra 生成一份计划，三局复用；JEV 为每个存活单位回答一道 Choice。小队共用一次请求，大队按预算分批，所有批次使用同一暂停中的观察。

默认 `realtime=False`、`step_mul=8`、seed 1、base 脚本对手；等待模型时实际游戏时钟保持不变。死亡单位由环境填 no-op，存活单位动作全部由 JEV 选择。原先的实时探测通过 `micro-realtime` 单独保留，未作为当前基线。

| 动作 | 含义 |
| --- | --- |
| 0 | 死亡单位 no-op，不提供给存活单位 |
| 1 | Stop，取消当前订单 |
| 2 / 3 / 4 / 5 | 北 / 南 / 东 / 西，各移动 2 个世界单位 |
| 6+j | 攻击可见敌人 e{j}；医疗机改为治疗友军 u{j} |

保留 SMAC 中心距离 6 的攻击候选限制，原生射程另行描述；技能、变形、任意坐标和目标预约没有加入动作空间。

四项 P0：公共地图背景；明确的射程和合法候选语义；分批中完整的队友核心字段；将 SC2 原生 ResponseAction 和观察错误回传到下一轮。命令被接受不等于已经造成伤害。

| 文件 | 职责 |
| --- | --- |
| [policy.py](jev_micro/policy.py) | 结构化状态、单位问题、批次与动作解码 |
| [planner.py](jev_micro/planner.py) | 每图 Astra 计划及复用 |
| [scenario.py](jev_micro/scenario.py) | 公开地图先验与静态地形 |
| [feedback.py](jev_micro/feedback.py) | SC2 原生命令和错误映射 |
| [run.py](jev_micro/run.py) | 完整对局、日志、时钟核对与 replay |
| [suite.py](jev_micro/suite.py) | 全图批测、失败保留与恢复 |
| [catalog.py](jev_micro/catalog.py) | 35 张地图及两个开发场景标记 |
| [skills/focus_fire.py](jev_micro/skills/focus_fire.py) | FocusFireBudget 集火伤害预算分配器 |
| [skills/cover_retreat.py](jev_micro/skills/cover_retreat.py) | CoverRetreat 危险识别、撤退方向与掩护分配 |
| [skills/heal_assign.py](jev_micro/skills/heal_assign.py) | HealAssign 多治疗者联合分配与持续治疗 |
| [skills/adaptive_focus_fire.py](jev_micro/skills/adaptive_focus_fire.py) | AdaptiveFocusFire 组合式自适应集火 |
| [skill_policy.py](jev_micro/skill_policy.py) | JEV 小队级技能选择问题与解码 |
| [skill_manager.py](jev_micro/skill_manager.py) | 技能生命周期、多步执行与终止 |

从仓库根目录启动全图三局：

```powershell
py -3.10 jev_star.py micro-suite --all-maps --policies jev --episodes 3 --seed 1 --step-mul 8 --planner codex --planner-model gpt-6-astra --planner-effort medium --planner-timeout 180 --request-timeout 15 --batch-concurrency 4 --api-retries 2 --output-dir jev_runs/my-sweep
```

恢复同一配置时增加 `--resume` 并使用相同输出目录。已完成的三局地图复用，失败尝试保留并创建新的重试目录；有效结果以汇总中的 `summary_file` 为准。切换接口、计划或参数应创建新的实验目录。

`SMAC_HARD_maps/` 集中保存全部 35 张注册地图，包括原上游旧目录中的地图。环境使用本目录自带且经修改的 `pysc2/`；不要用 pip 安装的另一份 PySC2 替代它。

结果见 [实验记录](../docs/experiments.md)，安装和密钥配置见 [总说明](../README.md#简体中文)。
