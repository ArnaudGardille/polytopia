import type { GameInfo } from '../types';

interface GameListProps {
  games: GameInfo[];
  selectedGameId: string | null;
  onSelectGame: (gameId: string) => void;
  isLoading?: boolean;
}

export function GameList({
  games,
  selectedGameId,
  onSelectGame,
  isLoading,
}: GameListProps) {
  if (isLoading) {
    return (
      <div className="p-8 text-center">
        <div className="text-gray-400">Chargement des replays...</div>
      </div>
    );
  }

  if (games.length === 0) {
    return (
      <div className="p-8 text-center">
        <div className="text-gray-400">Aucun replay disponible</div>
      </div>
    );
  }

  return (
    <div className="p-4">
      <h2 className="text-2xl font-bold mb-4 text-white">Replays disponibles</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {games.map((game) => {
          const isSelected = game.id === selectedGameId;
          return (
            <div
              key={game.id}
              onClick={() => onSelectGame(game.id)}
              className={`
                p-4 rounded-lg border-2 cursor-pointer transition-all
                ${
                  isSelected
                    ? 'border-blue-500 bg-blue-500/20'
                    : 'border-gray-700 bg-gray-800 hover:border-gray-600 hover:bg-gray-700'
                }
              `}
            >
              <div className="font-semibold text-white mb-2">{game.id}</div>
              <div className="text-sm text-gray-400 space-y-1">
                <div>
                  Carte: {game.metadata.width} × {game.metadata.height}
                </div>
                <div>Joueurs: {game.metadata.num_players}</div>
                {game.metadata.final_turn !== null && (
                  <div>Tours: {game.metadata.final_turn}</div>
                )}
                {game.metadata.game_done !== null && (
                  <div
                    className={
                      game.metadata.game_done
                        ? 'text-green-400'
                        : 'text-yellow-400'
                    }
                  >
                    {game.metadata.game_done ? '✓ Terminé' : 'En cours'}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

