import type { Screen, GameMode } from '../types';

interface ModeSelectionMenuProps {
  onNavigate: (screen: Screen, mode?: GameMode) => void;
}

export function ModeSelectionMenu({ onNavigate }: ModeSelectionMenuProps) {
  const handleModeSelect = (mode: GameMode) => {
    onNavigate('gameSetup', mode);
  };

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Background avec gradient low-poly */}
      <div className="absolute inset-0 bg-gradient-to-b from-pink-300 via-orange-200 to-yellow-100">
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
            onClick={() => onNavigate('mainMenu')}
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

        {/* Cartes de mode */}
        <div className="flex-1 flex flex-col items-center justify-center px-4 py-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl w-full mb-6">
            {/* PERFECTION */}
            <button
              onClick={() => handleModeSelect('perfection')}
              className="bg-blue-400 hover:bg-blue-500 rounded-2xl p-8 text-white transition-all transform hover:scale-105 shadow-xl flex flex-col items-center"
            >
              {/* Icône couronne */}
              <div className="mb-4">
                <svg
                  className="w-16 h-16 text-yellow-400"
                  fill="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path d="M5 16L3 5l5.5 5L12 4l3.5 6L21 5l-2 11H5zm14 3c0 .6-.4 1-1 1H6c-.6 0-1-.4-1-1v-1h14v1z" />
                </svg>
              </div>
              <h2 className="text-3xl font-bold mb-4">PERFECTION</h2>
              <p className="text-center text-sm opacity-90">
                Show your skills on the global hiscore in the classic 30 turns
                game.
              </p>
            </button>

            {/* DOMINATION */}
            <button
              onClick={() => handleModeSelect('domination')}
              className="bg-blue-400 hover:bg-blue-500 rounded-2xl p-8 text-white transition-all transform hover:scale-105 shadow-xl flex flex-col items-center"
            >
              {/* Icône crâne */}
              <div className="mb-4">
                <svg
                  className="w-16 h-16 text-gray-300"
                  fill="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path d="M12 2C8.13 2 5 5.13 5 9c0 2.38 1.19 4.47 3 5.74V17c0 .55.45 1 1 1h6c.55 0 1-.45 1-1v-2.26c1.81-1.27 3-3.36 3-5.74 0-3.87-3.13-7-7-7zm-2 12v-1h4v1h-4zm0-2v-1h4v1h-4z" />
                </svg>
              </div>
              <h2 className="text-3xl font-bold mb-4">DOMINATION</h2>
              <p className="text-center text-sm opacity-90">
                Play until there is only one tribe left, with no time limit.
              </p>
            </button>

            {/* GLORY */}
            <button
              onClick={() => handleModeSelect('glory')}
              className="bg-blue-400 hover:bg-blue-500 rounded-2xl p-8 text-white transition-all transform hover:scale-105 shadow-xl flex flex-col items-center"
            >
              {/* Icône étoile */}
              <div className="mb-4">
                <svg
                  className="w-16 h-16 text-yellow-300"
                  fill="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
                </svg>
              </div>
              <h2 className="text-3xl font-bold mb-4">GLORY</h2>
              <p className="text-center text-sm opacity-90">
                First player to reach 10,000 points wins.
              </p>
            </button>

            {/* MIGHT */}
            <button
              onClick={() => handleModeSelect('might')}
              className="bg-blue-400 hover:bg-blue-500 rounded-2xl p-8 text-white transition-all transform hover:scale-105 shadow-xl flex flex-col items-center"
            >
              {/* Icône épée */}
              <div className="mb-4">
                <svg
                  className="w-16 h-16 text-gray-200"
                  fill="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path d="M6.92 5.51L3.71 2.29 2.29 3.71l3.21 3.21c-1.1.9-1.8 2.2-1.8 3.68 0 2.49 2.01 4.5 4.5 4.5.48 0 .94-.08 1.38-.21l3.21 3.21 1.42-1.42-3.21-3.21c.13-.44.21-.9.21-1.38 0-1.48-.7-2.78-1.8-3.68zm8.58 8.58l-1.42 1.42 3.21 3.21c1.1-.9 1.8-2.2 1.8-3.68 0-2.49-2.01-4.5-4.5-4.5-.48 0-.94.08-1.38.21l-3.21-3.21-1.42 1.42 3.21 3.21c-.13.44-.21.9-.21 1.38 0 1.48.7 2.78 1.8 3.68l3.21 3.21z" />
                </svg>
              </div>
              <h2 className="text-3xl font-bold mb-4">MIGHT</h2>
              <p className="text-center text-sm opacity-90">
                Capture all enemy capitals to win.
              </p>
            </button>
          </div>

          {/* CREATIVE */}
          <button
            onClick={() => handleModeSelect('creative')}
            className="bg-blue-400 hover:bg-blue-500 rounded-2xl p-8 text-white transition-all transform hover:scale-105 shadow-xl flex flex-col items-center max-w-md w-full"
          >
            {/* Icône île avec maison */}
            <div className="mb-4 flex items-center justify-center">
              <div className="relative">
                {/* Île verte */}
                <div className="w-12 h-8 bg-green-500 rounded-t-full"></div>
                {/* Maison */}
                <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-2">
                  <div className="w-4 h-4 bg-white"></div>
                  <div className="w-6 h-3 bg-red-500 -mt-1"></div>
                </div>
                {/* Drapeau */}
                <div className="absolute top-0 right-0 transform translate-x-2 -translate-y-3">
                  <div className="w-1 h-4 bg-gray-400"></div>
                  <div className="w-3 h-2 bg-red-500 -mt-2 ml-1"></div>
                </div>
              </div>
            </div>
            <h2 className="text-3xl font-bold mb-4">CREATIVE</h2>
            <p className="text-center text-sm opacity-90">
              Set up your own game and play however you like.
            </p>
          </button>
        </div>
      </div>
    </div>
  );
}




