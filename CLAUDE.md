# CLAUDE.md

Guidance for AI assistants working in this repository. The project is a **Polytopia-inspired game engine in JAX** with FastAPI backend, React frontend, and RL wrappers.

The repository's primary documentation language is **French** (see `README.md`, `.cursorrules`). Match that style when editing existing docs/comments; code identifiers stay in English.

---

## 1. Architecture

Strict separation between layers — **never cross these boundaries**:

```
polytopia_jax/
├─ core/        # Pure JAX game engine (jit/vmap-compatible)
│  ├─ state.py     # GameState pytree (@struct.dataclass)
│  ├─ actions.py   # ActionType, encode/decode_action (30-bit packed int)
│  ├─ rules.py     # step(), legal_actions_mask(), unit/building stats
│  ├─ init.py      # init_random(), GameConfig
│  ├─ reward.py    # RL reward functions
│  └─ score.py     # Perfection-mode scoring
├─ ai/          # Heuristic strategies (rush/economy/random/idle)
│  └─ strategies.py
└─ web/         # FastAPI backend (no game logic)
   ├─ api.py            # HTTP endpoints
   ├─ models.py         # Pydantic GameStateView + serialization
   ├─ replay_store.py   # Replay JSON persistence
   ├─ live_game_store.py# Live session management
   ├─ serialize.py
   └─ view_options.py   # reveal_map / unlock_all_techs overrides

rl/                # Gym/PettingZoo wrappers (optional deps)
├─ session.py      # SimulationSession (drives core + AI opponents)
├─ gym_env.py      # Single-agent
└─ pettingzoo_env.py # Multi-agent (AEC/Parallel)

frontend/          # React + TypeScript + Vite + Tailwind, SVG board
scripts/           # Replay generation, web demo, wiki scrapers
tests/             # pytest suite mirroring source layout
replays/           # Generated game replays (JSON, gitignored except test_game.json)
live_games/        # Persisted live sessions (gitignored)
```

**Hard rules:**
- `core/` must stay pure JAX: no IO, no mutation, no imports from `web`/`ai`/`rl`/`frontend`.
- `frontend/` and `web/` must contain **zero** game logic — only display and serialization.
- `GameState` → JSON conversion happens **only** in `web/models.py` (via `GameStateView.from_raw_state`).
- TypeScript types in `frontend/src/types.ts` must mirror Pydantic models in `web/models.py`.

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
- Logical coordinates are aligned `(x, y)`, **not** offset-hex. The hex look is purely visual in `frontend/Board.tsx`.
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
- Always validate moves via `legal_actions_mask(state)` from `core/rules.py`.
- The mask must be computed in core — never re-derive legality in `web/` or `frontend/`.

### Units, cities, players
- Units use fixed-size arrays `[N_units_max]` with `units_active[i]` flag for liveness.
- Cities are dense grids `[H, W]`: presence = `city_owner[i,j] != NO_OWNER` and/or `city_level[i,j] > 0`.
- Player-indexed arrays (`player_stars`, `player_techs`, `player_score`, `score_*`, `player_income_bonus`) have shape `[num_players]` or `[num_players, num_techs]`.

### Score vs reward
- `core/score.py` → in-game scoring (Perfection mode breakdown: territory, population, military, resources, exploration).
- `core/reward.py` → RL training signal.
- Don't conflate the two.

---

## 3. Backend (`polytopia_jax/web/`)

FastAPI app at `polytopia_jax.web.api:app`. Key endpoints:

**Replays:**
- `GET /games` — list available replays
- `GET /games/{id}/metadata`
- `GET /games/{id}/replay` — full replay
- `GET /games/{id}/state/{turn}` — single state

**Live games:**
- `POST /live/perfection` | `/live/creative` | `/live/glory` | `/live/might` — create session (body: `LiveGameConfig` with `opponents`, `difficulty`, `seed`, `strategy`, plus mode-specific fields like `board_size`, `max_turns`)
- `GET /live/{game_id}` — current state
- `POST /live/{game_id}/action` — apply encoded action (body: `{"action_id": int}`)
- `POST /live/{game_id}/end_turn` — explicitly end human turn

**View options** (query params on most endpoints): `reveal_map`, `unlock_all_techs` — applied via `view_options.apply_view_overrides`.

CORS is open (`allow_origins=["*"]`) for dev. Live sessions persist to `live_games/` and are reloaded on startup.

When adding endpoints: keep them async, run blocking JAX work via `loop.run_in_executor(None, ...)` (see existing `post_live_action`), and convert to dict via `response.model_dump()` before returning to avoid Pydantic's slow JSON path.

---

## 4. Frontend (`frontend/`)

React 18 + TypeScript + Vite + Tailwind. All game state fetched from FastAPI — **never** simulate in the browser.

- Components: `App.tsx` (top-level navigation state machine), `Board.tsx` (SVG rendering with hex visual), `HUD.tsx`, `LiveGameView.tsx`, `GameList.tsx`, `MainMenu.tsx`, `ModeSelectionMenu.tsx`, `GameSetupMenu.tsx`, `Scoreboard.tsx`.
- API client: `frontend/src/api.ts` — single source of truth for HTTP calls.
- Types: `frontend/src/types.ts` — keep in sync with `web/models.py`.
- Static data: `frontend/src/data/{units,buildings,techTree,resources}.ts`.
- Utils: `actionEncoder.ts` mirrors the bit layout from `core/actions.py` — keep in lockstep.
- Dev proxy: `vite.config.ts` forwards `/games` to `http://localhost:8000`. For other paths (e.g. `/live/...`), use the env var `VITE_API_URL` or extend the proxy.
- Persisted state: `localStorage['polytopia:lastLiveGameId']` for resume.

Conventions: functional components only, hooks for state, types live in `types.ts` (no inline `interface`s).

---

## 5. RL wrappers (`rl/`)

`gymnasium` / `pettingzoo` are **optional** dependencies — imports are conditional. `SimulationSession` (in `rl/session.py`) is the shared driver: it owns the `GameState`, runs AI opponents to completion before returning control to player 0, and applies difficulty bonuses (`star_bonus` per `DIFFICULTY_PRESETS`).

`SimulationConfig` defaults: `height=12, width=12, opponents=1, max_units=64, max_turns=30, difficulty="normal", ai_strategy="rush", game_mode=DOMINATION`. `ai_strategy` accepts `rush | economy | random | idle`.

---

## 6. Naming conventions

| Kind             | Convention             | Example                  |
|------------------|------------------------|--------------------------|
| Python class     | PascalCase             | `GameState`, `ActionType`|
| Python function  | snake_case             | `encode_action`          |
| Python constant  | UPPER_SNAKE_CASE       | `MAX_UNIT_TYPES`, `NO_OWNER` |
| Python file      | snake_case             | `state.py`               |
| TS component     | PascalCase             | `Board.tsx`              |
| TS function      | camelCase              | `fetchGameState`         |
| TS type/interface| PascalCase             | `GameStateView`          |
| TS constant      | UPPER_SNAKE_CASE / `as const` | `TerrainType`     |

---

## 7. Development workflows

### Backend
```bash
source venv/bin/activate            # virtualenv at venv/
pip install -e .                    # install in editable mode
uvicorn polytopia_jax.web.api:app --reload --host 0.0.0.0 --port 8000
# or
python scripts/run_web_demo.py
```

API docs: http://localhost:8000/docs

### Frontend
```bash
cd frontend
npm install
npm run dev          # http://localhost:5173 (proxies /games to :8000)
npm run build        # tsc + vite build
npm run lint         # ESLint, --max-warnings 0
```

### Tests (pytest)
```bash
pytest tests/ -v                    # full suite
pytest tests/test_rules.py -v       # specific module
pytest tests/test_web/ -v
pytest tests/test_ai/ -v
pytest tests/test_rl/ -v
```

Test config in `pyproject.toml`: `testpaths = ["tests"]`, files `test_*.py`, classes `Test*`, functions `test_*`. Test fixtures live in `tests/test_web/fixtures/`.

### Replay generation
```bash
python scripts/generate_replay.py --output replays/game_001.json
```

### Other useful scripts
- `scripts/download_polytopia_images.py` / `scrape_wiki.py` — asset pipeline
- `scripts/crop_terrain_images.py`, `scripts/generate_manifest_icons.py`
- `scripts/simple_bot.py` — minimal bot baseline

---

## 8. Game development phases

The codebase grows phase by phase. When extending, check status and document limitations in docstrings:

- **Phase 0–6** ✅ — Stable economic loop, scoring, tech tree, ground unit variety (Warrior, Defender, Archer, Rider, Knight, Swordsman, Catapult, Giant), Raft naval, AI strategies + difficulties, all merged.
- **Phase 7** 🚧 — Monuments, temples, special tribes (Polaris/Cymanti/Aquarion). Configuration toggles via `web/api.py` + `frontend/`.

Game modes implemented: `DOMINATION`, `PERFECTION`, `CREATIVE`, `GLORY`, `MIGHT` (see `GameMode` IntEnum). Live endpoints exist for each.

---

## 9. Editing rules of thumb

1. **Never** import from `web`/`frontend`/`rl` inside `core/`.
2. **Never** add IO (file/network/random RNG state) to `core/` — pass keys as args.
3. When you add a `GameState` field, update **all** of: `state.py` dataclass, `state.py::create_empty`, `init.py::init_random`, the relevant rules transitions, `web/serialize.py`, `web/models.py::GameStateView`, and `frontend/src/types.ts`.
4. When you add a unit, extend every `UNIT_*` array in `rules.py` to length `MAX_UNIT_TYPES` and add the enum value to `UnitType`. Same for techs (`TechType` + `TECH_COST` + dependencies).
5. When you add an action type, bump `ActionType.NUM_ACTIONS`, extend `legal_actions_mask`, handle it in `step`, and update both `actionEncoder.ts` and any AI strategy that should use it.
6. Run `pytest tests/ -v` before committing — the suite is the source of truth for expected mechanics.
7. Branch policy: develop on the branch declared by the harness (currently `claude/add-claude-documentation-nbGtD`); never push to `main` without explicit approval.

---

## 10. Reference docs in repo

- `README.md` — project overview and roadmap (French)
- `SETUP.md` — venv + first-run instructions
- `frontend/README.md` — frontend status and screen flow
- `frontend/AGENCEMENT_TERRAIN.md` — terrain layout notes
- `Polytopia.md`, `COMPARAISON_POLYTOPIA.md`, `COMPTE_RENDU_*.md` — design specs and gap analyses vs the original game
- `.cursorrules` — Cursor-specific conventions (kept consistent with this file)
- `scripts/GUIDE_INSTALLATION.md`, `scripts/README_SCRAPER.md`, `scripts/README_DOWNLOAD_IMAGES.md` — asset/scraping pipelines
