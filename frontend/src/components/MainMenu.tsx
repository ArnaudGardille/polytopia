import type { Screen } from '../types';

interface MainMenuProps {
  onNavigate: (screen: Screen) => void;
  onResumeGame: () => void;
  canResume: boolean;
  isResuming?: boolean;
  resumeError?: string | null;
}

export function MainMenu({
  onNavigate,
  onResumeGame,
  canResume,
  isResuming = false,
  resumeError,
}: MainMenuProps) {
  const handleNewGame = () => {
    onNavigate('modeSelection');
  };

  const handleMultiplayer = () => {
    // TODO: Implémenter le multijoueur
    console.log('Multiplayer');
  };

  const handleReplay = () => {
    onNavigate('game');
  };

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Background avec gradient low-poly */}
      <div className="absolute inset-0 bg-gradient-to-b from-pink-400 via-orange-300 to-peach-200">
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
        {/* Header avec NEWS */}
        <div className="flex justify-end p-6">
          <button className="bg-blue-400 hover:bg-blue-500 text-white rounded-full px-6 py-2 flex items-center gap-2 transition-colors">
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9"
              />
            </svg>
            <span className="font-semibold">NEWS</span>
          </button>
        </div>

        {/* Titre et icône */}
        <div className="flex-1 flex flex-col items-center justify-center px-4">
          {/* Icône 3D */}
          <div className="mb-4">
            <div className="w-16 h-16 bg-white rounded-lg shadow-lg flex items-center justify-center">
              <div className="w-12 h-12 bg-green-500 rounded"></div>
            </div>
          </div>

          {/* Titre */}
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-12 text-center tracking-wide">
            THE BATTLE OF POLYTOPIA
          </h1>

          {/* Boutons principaux */}
          <div className="flex flex-col gap-4 w-full max-w-md">
            <button
              onClick={handleNewGame}
              className="bg-blue-400 hover:bg-blue-500 text-white font-semibold py-4 px-8 rounded-xl text-lg transition-all transform hover:scale-105 shadow-lg"
            >
              NEW GAME
            </button>
            <button
              onClick={onResumeGame}
              disabled={!canResume || isResuming}
              className={`bg-blue-400 text-white font-semibold py-4 px-8 rounded-xl text-lg transition-all transform shadow-lg ${
                !canResume || isResuming
                  ? 'opacity-60 cursor-not-allowed'
                  : 'hover:bg-blue-500 hover:scale-105'
              }`}
            >
              {isResuming ? 'RESUME EN COURS...' : 'RESUME GAME'}
            </button>
            <button
              onClick={handleMultiplayer}
              className="bg-blue-400 hover:bg-blue-500 text-white font-semibold py-4 px-8 rounded-xl text-lg transition-all transform hover:scale-105 shadow-lg"
            >
              MULTIPLAYER
            </button>
            <button
              onClick={handleReplay}
              className="bg-blue-400 hover:bg-blue-500 text-white font-semibold py-4 px-8 rounded-xl text-lg transition-all transform hover:scale-105 shadow-lg"
            >
              REPLAY
            </button>
            {resumeError && (
              <div className="mt-2 text-center text-sm font-semibold text-red-800 bg-white/70 rounded-lg py-2 px-3">
                {resumeError}
              </div>
            )}
          </div>
        </div>

        {/* Barre de navigation en bas */}
        <div className="pb-8 px-4">
          <div className="flex justify-center gap-8 md:gap-12">
            {/* Settings */}
            <button className="flex flex-col items-center gap-2 text-white hover:opacity-80 transition-opacity">
              <div className="w-12 h-12 rounded-full border-2 border-white flex items-center justify-center">
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                </svg>
              </div>
              <span className="text-sm font-medium">Settings</span>
            </button>

            {/* High Score */}
            <button className="flex flex-col items-center gap-2 text-white hover:opacity-80 transition-opacity">
              <div className="w-12 h-12 rounded-full border-2 border-white flex items-center justify-center">
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z"
                  />
                </svg>
              </div>
              <span className="text-sm font-medium">High Score</span>
            </button>

            {/* Throne Room */}
            <button className="flex flex-col items-center gap-2 text-white hover:opacity-80 transition-opacity">
              <div className="w-12 h-12 rounded-full border-2 border-white flex items-center justify-center">
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z"
                  />
                </svg>
              </div>
              <span className="text-sm font-medium">Throne Room</span>
            </button>

            {/* About */}
            <button className="flex flex-col items-center gap-2 text-white hover:opacity-80 transition-opacity">
              <div className="w-12 h-12 rounded-full border-2 border-white flex items-center justify-center">
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"
                  />
                </svg>
              </div>
              <span className="text-sm font-medium">About</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

