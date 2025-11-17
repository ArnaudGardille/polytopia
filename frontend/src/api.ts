import type {
  GamesListResponse,
  ReplayResponse,
  StateResponse,
  GameInfo,
} from './types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function fetchAPI<T>(endpoint: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`);
  if (!response.ok) {
    throw new Error(`API Error: ${response.status} ${response.statusText}`);
  }
  return response.json();
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

