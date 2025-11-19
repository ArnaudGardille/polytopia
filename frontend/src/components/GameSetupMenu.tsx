import { useState } from 'react';
import type { Screen, GameMode, Difficulty, GameConfig } from '../types';

interface GameSetupMenuProps {
  mode: GameMode;
  onNavigate: (screen: Screen, config?: GameConfig) => void;
  onStartGame?: (config: GameConfig) => void | Promise<void>;
  isStarting?: boolean;
  startError?: string | null;
}

export function GameSetupMenu({
  mode,
  onNavigate,
  onStartGame,
  isStarting,
  startError,
}: GameSetupMenuProps) {
  const [opponents, setOpponents] = useState(9);
  const [difficulty, setDifficulty] = useState<Difficulty>('crazy');

  // Calculer les paramètres de la carte basés sur le nombre d'opposants
  const calculateMapSize = (numOpponents: number): number => {
    // Formule approximative : plus d'opposants = plus grande carte
    const baseSize = 128;
    const sizePerOpponent = 16;
    return baseSize + numOpponents * sizePerOpponent;
  };

  const mapSize = calculateMapSize(opponents);
  const turnsLimit = mode === 'perfection' ? 30 : null;

  const handleStartGame = () => {
    const config: GameConfig = {
      mode,
      opponents,
      difficulty,
    };
    if (onStartGame) {
      onStartGame(config);
    } else {
      console.log('Starting game with config:', config);
    }
  };

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Background avec gradient low-poly rose/pourpre */}
      <div className="absolute inset-0 bg-gradient-to-b from-pink-300 via-pink-400 to-purple-500">
        {/* Particules/stars */}
        <div className="absolute inset-0">
          {Array.from({ length: 50 }).map((_, i) => (
            <div
              key={i}
              className="absolute w-1 h-1 bg-white rounded-full opacity-60"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
              }}
            />
          ))}
        </div>
      </div>

      {/* Contenu principal */}
      <div className="relative z-10 min-h-screen flex flex-col">
        {/* Bouton retour */}
        <div className="p-6">
          <button
            onClick={() => onNavigate('modeSelection')}
            className="bg-gray-200 hover:bg-gray-300 rounded-full w-12 h-12 flex items-center justify-center transition-colors shadow-lg"
          >
            <svg
              className="w-6 h-6 text-gray-700"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 19l-7-7 7-7"
              />
            </svg>
          </button>
        </div>

        {/* Section Game Setup */}
        <div className="flex-1 flex items-center justify-center px-4 py-8">
          <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-8 max-w-2xl w-full border-2 border-white/20 shadow-2xl">
            <h2 className="text-3xl font-bold text-white mb-8">Game Setup</h2>

            {/* Sélection d'opposants */}
            <div className="mb-8">
              <label className="block text-white text-xl font-semibold mb-4">
                Opponents
              </label>
              <div className="flex gap-2 flex-wrap">
                {[3, 4, 5, 6, 7, 8, 9].map((num) => (
                  <button
                    key={num}
                    onClick={() => setOpponents(num)}
                    className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                      opponents === num
                        ? 'bg-white text-pink-600 shadow-lg scale-105'
                        : 'bg-white/20 text-white hover:bg-white/30'
                    }`}
                  >
                    {num}
                  </button>
                ))}
              </div>
            </div>

            {/* Sélection de difficulté */}
            <div className="mb-8">
              <label className="block text-white text-xl font-semibold mb-4">
                Difficulty
              </label>
              <div className="flex gap-2 flex-wrap">
                {(['easy', 'normal', 'hard', 'crazy'] as Difficulty[]).map(
                  (diff) => (
                    <button
                      key={diff}
                      onClick={() => setDifficulty(diff)}
                      className={`px-6 py-3 rounded-lg font-semibold transition-all capitalize ${
                        difficulty === diff
                          ? 'bg-white text-pink-600 shadow-lg scale-105'
                          : 'bg-white/20 text-white hover:bg-white/30'
                      }`}
                    >
                      {diff}
                    </button>
                  )
                )}
              </div>
            </div>

            {/* Affichage des paramètres */}
            <div className="mb-8 p-4 bg-white/10 rounded-lg">
              <p className="text-white text-sm">
                {opponents} opponents, {mapSize} tiles map
                {turnsLimit && `, ${turnsLimit} turns limit`}
              </p>
            </div>
          </div>
        </div>

        {/* Bouton START GAME */}
        <div className="pb-8 px-4 flex justify-center">
          <div className="flex flex-col items-center gap-3">
            {startError && (
              <p className="text-red-200 text-sm bg-red-500/30 px-4 py-2 rounded-lg">
                {startError}
              </p>
            )}
            <button
              onClick={handleStartGame}
              disabled={!!isStarting}
              className="bg-blue-500 hover:bg-blue-600 disabled:opacity-60 disabled:cursor-not-allowed text-white font-bold py-4 px-12 rounded-xl text-xl transition-all transform hover:scale-105 shadow-xl"
            >
              {isStarting ? 'Chargement...' : 'START GAME'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}


