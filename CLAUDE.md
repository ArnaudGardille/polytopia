# CLAUDE.md

Guidance for AI assistants working in this repository. The project is a **Polytopia-inspired game engine in JAX** designed to **synchronize AI agents** (RL training, self-play). There is no human-facing web/UI layer.

The repository's primary documentation language is **French** (see `README.md`). Match that style when editing existing docs/comments; code identifiers stay in English.

---

## 1. Architecture

```
polytopia_jax/
├─ core/        # Pure JAX game engine (jit/vmap-compatible)
│  ├─ state.py     # GameState pytree (@struct.dataclass)
│  ├─ actions.py   # ActionType, encode/decode_action (30-bit packed int)
│  ├─ rules.py     # step(), legal_actions_mask(), unit/building stats
│  ├─ init.py      # init_random(), GameConfig
│  ├─ reward.py    # RL reward functions
│  └─ score.py     # Perfection-mode scoring
└─ ai/          # Heuristic strategies (rush/economy/random/idle)
   └─ strategies.py

rl/                # SimulationSession driver (gymnasium/pettingzoo are optional deps)
└─ session.py

tests/             # pytest suite mirroring source layout
```

**Hard rules:**
- `core/` must stay pure JAX: no IO, no mutation, no imports from `ai`/`rl`.
- `core/` does not use global RNG state — all randomness flows through `key` arguments.

---

## 2. Core engine conventions

### JAX / Flax
- Use `jax.numpy as jnp` everywhere in `core/` (never plain `numpy`).
- State structs are `@struct.dataclass` from `flax.struct` — **not** stdlib `@dataclass`.
- All `GameState` fields are JAX arrays (so `jit`/`vmap` work). Dimensions (`height`, `width`, `max_units`, `num_players`) are `int` because they're static.
- Never mutate arrays — use `.at[...].set(...)` / `state.replace(field=...)`.
- For conditionals inside traced code, use `jax.lax.cond` / `jnp.where`.

### Enums and constants
- Type enums (`UnitType`, `TerrainType`, `TechType`, `ActionType`, `Direction`, `GameMode`, `MapSize`, `MapType`, `ResourceType`, `BuildingType`) are `IntEnum`s and **always** end with a `NUM_*` sentinel for loop bounds.
- `NO_OWNER = -1` for empty city/unit ownership; `city_level = 0` means no city.
- Unit stats are stored as padded JAX arrays of length `MAX_UNIT_TYPES = 16` (see `rules.py`: `UNIT_HP_MAX`, `UNIT_ATTACK`, `UNIT_DEFENSE`, `UNIT_MOVEMENT`, `UNIT_COST`, `UNIT_ATTACK_RANGE`, `UNIT_REQUIRED_TECH`, `UNIT_IS_NAVAL`, `UNIT_CAN_ENTER_SHALLOW/DEEP`, `UNIT_CAN_PROMOTE`). When adding a unit type, extend **all** of these arrays consistently.

### Grid
- Logical coordinates are aligned `(x, y)`, **not** offset-hex.
- 8 movement directions with deltas in `{-1, 0, 1}` × `{-1, 0, 1}` — see `DIRECTION_DELTA` in `actions.py`.

### Action encoding
Actions are packed into a single 30-bit int (fits in `int32`):

| Field        | Bits  | Range     |
|--------------|-------|-----------|
| action_type  | 0–3   | 0–15      |
| unit_id      | 4–11  | 0–255     |
| direction    | 12–14 | 0–7       |
| target_x     | 15–19 | 0–31      |
| target_y     | 20–24 | 0–31      |
| unit_type    | 25–29 | 0–31      |

Use `encode_action(...)` / `decode_action(...)`. `decode_action` works in both Python and traced contexts — preserve that dual-mode behavior.

`ActionType` values: `NO_OP`, `MOVE`, `ATTACK`, `TRAIN_UNIT`, `BUILD`, `RESEARCH_TECH`, `END_TURN`, `HARVEST_RESOURCE`, `RECOVER`.

### Action masking
Always validate moves via `legal_actions_mask(state)` from `core/rules.py`. The mask must be computed in core.

### Units, cities, players
- Units use fixed-size arrays `[N_units_max]` with `units_active[i]` flag for liveness.
- Cities are dense grids `[H, W]`: presence = `city_owner[i,j] != NO_OWNER` and/or `city_level[i,j] > 0`.
- Player-indexed arrays (`player_stars`, `player_techs`, `player_score`, `score_*`, `player_income_bonus`) have shape `[num_players]` or `[num_players, num_techs]`.

### Score vs reward
- `core/score.py` → in-game scoring (Perfection mode breakdown).
- `core/reward.py` → RL training signal.
- Don't conflate the two.

---

## 3. RL driver (`rl/session.py`)

`SimulationSession` is the shared driver: it owns the `GameState`, runs AI opponents to completion before returning control to player 0, and applies difficulty bonuses (`star_bonus` per `DIFFICULTY_PRESETS`).

`SimulationConfig` defaults: `height=12, width=12, opponents=1, max_units=64, max_turns=30, difficulty="normal", ai_strategy="rush", game_mode=DOMINATION`. `ai_strategy` accepts `rush | economy | random | idle`.

`gymnasium` and `pettingzoo` are **optional** dependencies; install via `pip install -e .[rl]`.

---

## 4. Naming conventions

| Kind             | Convention             | Example                  |
|------------------|------------------------|--------------------------|
| Python class     | PascalCase             | `GameState`, `ActionType`|
| Python function  | snake_case             | `encode_action`          |
| Python constant  | UPPER_SNAKE_CASE       | `MAX_UNIT_TYPES`, `NO_OWNER` |
| Python file      | snake_case             | `state.py`               |

---

## 5. Development workflows

```bash
python -m venv venv
source venv/bin/activate
pip install -e .[dev]
pytest tests/ -v
```

Specific tests:
```bash
pytest tests/test_rules.py -v
pytest tests/test_ai/ -v
pytest tests/test_rl/ -v
```

Test config in `pyproject.toml`: `testpaths = ["tests"]`, files `test_*.py`, classes `Test*`, functions `test_*`.

---

## 6. Editing rules of thumb

1. **Never** import IO (file/network/global RNG state) from `core/`.
2. When you add a `GameState` field, update `state.py` dataclass, `state.py::create_empty`, `init.py::init_random`, and the relevant rules transitions.
3. When you add a unit, extend every `UNIT_*` array in `rules.py` to length `MAX_UNIT_TYPES` and add the enum value to `UnitType`. Same for techs (`TechType` + `TECH_COST` + dependencies).
4. When you add an action type, bump `ActionType.NUM_ACTIONS`, extend `legal_actions_mask`, and handle it in `step`.
5. Run `pytest tests/ -v` before committing — the suite is the source of truth for expected mechanics.
6. Branch policy: develop on the branch declared by the harness; never push to `main` without explicit approval.
