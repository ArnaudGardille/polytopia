// Types correspondant aux modèles Pydantic du backend

export interface UnitView {
  id?: number;
  type: number;
  pos: [number, number];
  hp: number;
  owner: number;
  has_acted?: boolean;
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
  city_population?: number[][];
  city_ports?: number[][];
  resource_type?: number[][];
  resource_available?: boolean[][];
  player_stars?: number[];
  player_income?: number[];
  player_score?: number[];
  score_breakdown?: Record<string, number[]>;
  player_techs?: number[][];
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
  PLAIN_FRUIT: 5,
  FOREST_WITH_WILD_ANIMAL: 6,
  MOUNTAIN_WITH_MINE: 7,
  WATER_SHALLOW_WITH_FISH: 8,
} as const;

// Constantes d'unités (correspondant à UnitType dans le backend)
export const UnitType = {
  NONE: 0,
  WARRIOR: 1,
} as const;

export const ResourceType = {
  NONE: 0,
  FRUIT: 1,
  FISH: 2,
  ORE: 3,
} as const;

export const TechType = {
  NONE: 0,
  CLIMBING: 1,
  SAILING: 2,
  MINING: 3,
} as const;

// Types pour la navigation et la configuration de jeu
export type Screen = 'mainMenu' | 'modeSelection' | 'gameSetup' | 'game' | 'liveGame';

export type GameMode = 'perfection' | 'domination' | 'creative';

export type Difficulty = 'easy' | 'normal' | 'hard' | 'crazy';

export interface GameConfig {
  mode: GameMode;
  opponents: number; // 3-9
  difficulty: Difficulty;
}

// Types pour le mode live Perfection
export interface LiveGameHandle {
  gameId: string;
  maxTurns: number;
  opponents: number;
  difficulty: Difficulty | string;
}

export interface LiveGameStateResponse extends LiveGameHandle {
  state: GameStateView;
}

export interface LiveActionPayload {
  actionId: number;
}
