import type {
  GamesListResponse,
  ReplayResponse,
  StateResponse,
  GameInfo,
  GameStateView,
  Difficulty,
  LiveGameStateResponse,
} from './types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
    if (!response.ok) {
      let errorDetail = `${response.status} ${response.statusText}`;
      try {
        const errorData = await response.json();
        if (errorData.detail) {
          errorDetail = errorData.detail;
        }
      } catch {
        // Ignore JSON parse errors
      }
      throw new Error(`API Error: ${errorDetail}`);
    }
    return response.json();
  } catch (error) {
    if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
      throw new Error('Impossible de contacter le serveur. Vérifiez que le backend est lancé.');
    }
    throw error;
  }
}

/**
 * Liste tous les replays disponibles
 */
export async function listGames(): Promise<GamesListResponse> {
  return fetchAPI<GamesListResponse>('/games');
}

/**
 * Récupère le replay complet d'une partie
 */
export async function getReplay(gameId: string): Promise<ReplayResponse> {
  return fetchAPI<ReplayResponse>(`/games/${gameId}/replay`);
}

/**
 * Récupère l'état du jeu à un tour spécifique
 */
export async function getStateAtTurn(
  gameId: string,
  turn: number
): Promise<StateResponse> {
  return fetchAPI<StateResponse>(`/games/${gameId}/state/${turn}`);
}

/**
 * Récupère uniquement les métadonnées d'une partie
 */
export async function getMetadata(gameId: string): Promise<GameInfo> {
  const data = await fetchAPI<{ id: string; metadata: any }>(
    `/games/${gameId}/metadata`
  );
  return {
    id: data.id,
    metadata: data.metadata,
  };
}

interface LiveGameApiResponse {
  game_id: string;
  max_turns: number;
  opponents: number;
  difficulty: string;
  strategy: string;
  state: GameStateView;
}

function mapLiveResponse(data: LiveGameApiResponse): LiveGameStateResponse {
  return {
    gameId: data.game_id,
    maxTurns: data.max_turns,
    opponents: data.opponents,
    difficulty: data.difficulty,
    strategy: data.strategy,
    state: data.state,
  };
}

interface CreateGamePayload {
  opponents: number;
  difficulty: Difficulty;
  strategy?: string;
  seed?: number;
  boardSize?: number;
  maxTurns?: number;
}

export async function createPerfectionGame(
  payload: CreateGamePayload
): Promise<LiveGameStateResponse> {
  const body: Record<string, unknown> = {
    opponents: payload.opponents,
    difficulty: payload.difficulty,
  };
  if (payload.seed !== undefined) {
    body.seed = payload.seed;
  }
  if (payload.strategy) {
    body.strategy = payload.strategy;
  }

  const data = await fetchAPI<LiveGameApiResponse>('/live/perfection', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return mapLiveResponse(data);
}

export async function createCreativeGame(
  payload: CreateGamePayload
): Promise<LiveGameStateResponse> {
  const body: Record<string, unknown> = {
    opponents: payload.opponents,
    difficulty: payload.difficulty,
  };
  if (payload.seed !== undefined) {
    body.seed = payload.seed;
  }
  if (payload.strategy) {
    body.strategy = payload.strategy;
  }
  if (payload.boardSize !== undefined) {
    body.board_size = payload.boardSize;
  }
  if (payload.maxTurns !== undefined) {
    body.max_turns = payload.maxTurns;
  }

  const data = await fetchAPI<LiveGameApiResponse>('/live/creative', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return mapLiveResponse(data);
}

export async function createGloryGame(
  payload: CreateGamePayload
): Promise<LiveGameStateResponse> {
  const body: Record<string, unknown> = {
    opponents: payload.opponents,
    difficulty: payload.difficulty,
  };
  if (payload.seed !== undefined) {
    body.seed = payload.seed;
  }
  if (payload.strategy) {
    body.strategy = payload.strategy;
  }

  const data = await fetchAPI<LiveGameApiResponse>('/live/glory', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return mapLiveResponse(data);
}

export async function createMightGame(
  payload: CreateGamePayload
): Promise<LiveGameStateResponse> {
  const body: Record<string, unknown> = {
    opponents: payload.opponents,
    difficulty: payload.difficulty,
  };
  if (payload.seed !== undefined) {
    body.seed = payload.seed;
  }
  if (payload.strategy) {
    body.strategy = payload.strategy;
  }

  const data = await fetchAPI<LiveGameApiResponse>('/live/might', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return mapLiveResponse(data);
}

export async function getLiveGameState(
  gameId: string
): Promise<LiveGameStateResponse> {
  const data = await fetchAPI<LiveGameApiResponse>(`/live/${gameId}`);
  return mapLiveResponse(data);
}

export async function sendLiveAction(
  gameId: string,
  actionId: number
): Promise<LiveGameStateResponse> {
  console.log(`[API] Envoi action ${actionId} vers /live/${gameId}/action`);
  const data = await fetchAPI<LiveGameApiResponse>(`/live/${gameId}/action`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action_id: actionId }),
  });
  return mapLiveResponse(data);
}

export async function endLiveTurn(
  gameId: string
): Promise<LiveGameStateResponse> {
  const data = await fetchAPI<LiveGameApiResponse>(`/live/${gameId}/end_turn`, {
    method: 'POST',
  });
  return mapLiveResponse(data);
}
