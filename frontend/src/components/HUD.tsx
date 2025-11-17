import { getPlayerColor } from '../utils/iconMapper';
import type { GameStateView, ReplayMetadata } from '../types';

interface HUDProps {
  state: GameStateView;
  metadata?: ReplayMetadata;
  currentTurn: number;
  maxTurn: number;
  onPreviousTurn: () => void;
  onNextTurn: () => void;
  onPlayPause: () => void;
  isPlaying: boolean;
}

export function HUD({
  state,
  metadata,
  currentTurn,
  maxTurn,
  onPreviousTurn,
  onNextTurn,
  onPlayPause,
  isPlaying,
}: HUDProps) {
  const playerColor = getPlayerColor(state.current_player);
  const progress = maxTurn > 0 ? (currentTurn / maxTurn) * 100 : 0;

  return (
    <div className="bg-gray-800 text-white p-4 shadow-lg">
      <div className="max-w-7xl mx-auto">
        {/* Informations principales */}
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div className="flex items-center gap-4">
            <div>
              <span className="text-gray-400 text-sm">Tour</span>
              <div className="text-2xl font-bold">{state.turn}</div>
            </div>
            <div className="h-12 w-px bg-gray-600"></div>
            <div>
              <span className="text-gray-400 text-sm">Joueur actif</span>
              <div className="flex items-center gap-2">
                <div
                  className="w-6 h-6 rounded-full border-2 border-white"
                  style={{ backgroundColor: playerColor }}
                ></div>
                <span className="text-xl font-semibold">
                  Joueur {state.current_player + 1}
                </span>
              </div>
            </div>
            {state.done && (
              <div className="ml-4 px-3 py-1 bg-red-600 rounded-full text-sm font-semibold">
                Partie terminée
              </div>
            )}
          </div>

          {/* Métadonnées si disponibles */}
          {metadata && (
            <div className="text-sm text-gray-400">
              <div>
                Carte: {metadata.width} × {metadata.height}
              </div>
              <div>Joueurs: {metadata.num_players}</div>
            </div>
          )}
        </div>

        {/* Barre de progression */}
        <div className="mb-4">
          <div className="flex items-center justify-between text-sm text-gray-400 mb-1">
            <span>Tour {currentTurn}</span>
            <span>Tour {maxTurn}</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div
              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        </div>

        {/* Contrôles de navigation */}
        <div className="flex items-center justify-center gap-4">
          <button
            onClick={onPreviousTurn}
            disabled={currentTurn === 0}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-semibold transition-colors"
          >
            ← Précédent
          </button>
          <button
            onClick={onPlayPause}
            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold transition-colors"
          >
            {isPlaying ? '⏸ Pause' : '▶ Play'}
          </button>
          <button
            onClick={onNextTurn}
            disabled={currentTurn >= maxTurn}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-semibold transition-colors"
          >
            Suivant →
          </button>
        </div>
      </div>
    </div>
  );
}

