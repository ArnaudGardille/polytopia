// Types correspondant aux modèles Pydantic du backend

export interface UnitView {
  type: number;
  pos: [number, number];
  hp: number;
  owner: number;
}

export interface CityView {
  owner: number;
  level: number;
  pos: [number, number];
}

export interface GameStateView {
  terrain: number[][];
  cities: CityView[];
  units: UnitView[];
  current_player: number;
  turn: number;
  done: boolean;
}

export interface ReplayMetadata {
  height: number;
  width: number;
  num_players: number;
  max_turns?: number | null;
  seed?: number | null;
  final_turn?: number | null;
  total_actions?: number | null;
  game_done?: boolean | null;
}

export interface GameInfo {
  id: string;
  metadata: ReplayMetadata;
}

export interface ReplayResponse {
  id: string;
  metadata: ReplayMetadata;
  states: GameStateView[];
}

export interface StateResponse {
  turn: number;
  state: GameStateView;
}

export interface GamesListResponse {
  games: GameInfo[];
}

// Constantes de terrain (correspondant à TerrainType dans le backend)
export const TerrainType = {
  PLAIN: 0,
  FOREST: 1,
  MOUNTAIN: 2,
  WATER_SHALLOW: 3,
  WATER_DEEP: 4,
} as const;

// Constantes d'unités (correspondant à UnitType dans le backend)
export const UnitType = {
  NONE: 0,
  WARRIOR: 1,
} as const;

