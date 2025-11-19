import { useState, useEffect, useCallback } from 'react';
import { Board } from './components/Board';
import { HUD } from './components/HUD';
import { GameList } from './components/GameList';
import { MainMenu } from './components/MainMenu';
import { ModeSelectionMenu } from './components/ModeSelectionMenu';
import { GameSetupMenu } from './components/GameSetupMenu';
import { LiveGameView } from './components/LiveGameView';
import {
  listGames,
  getReplay,
  createPerfectionGame,
  sendLiveAction,
  endLiveTurn,
  getLiveGameState,
} from './api';
import type {
  GameInfo,
  GameStateView,
  ReplayResponse,
  Screen,
  GameMode,
  GameConfig,
  LiveGameStateResponse,
} from './types';
import './styles/App.css';

function App() {
  // Navigation
  const [currentScreen, setCurrentScreen] = useState<Screen>('mainMenu');
  const [selectedMode, setSelectedMode] = useState<GameMode | null>(null);

  // Replay viewer (pour l'écran "game")
  const [games, setGames] = useState<GameInfo[]>([]);
  const [selectedGameId, setSelectedGameId] = useState<string | null>(null);
  const [replay, setReplay] = useState<ReplayResponse | null>(null);
  const [currentTurn, setCurrentTurn] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Live game state
  const [liveSession, setLiveSession] = useState<LiveGameStateResponse | null>(null);
  const [liveSelectedUnitId, setLiveSelectedUnitId] = useState<number | null>(null);
  const [isStartingLiveGame, setIsStartingLiveGame] = useState(false);
  const [liveStartError, setLiveStartError] = useState<string | null>(null);
  const [liveRuntimeError, setLiveRuntimeError] = useState<string | null>(null);
  const [isLiveBusy, setIsLiveBusy] = useState(false);

  // Charger la liste des replays (seulement pour l'écran game)
  useEffect(() => {
    if (currentScreen !== 'game') return;

    async function loadGames() {
      try {
        setIsLoading(true);
        const response = await listGames();
        setGames(response.games);
        // Sélectionner le premier replay par défaut
        if (response.games.length > 0) {
          setSelectedGameId(response.games[0].id);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Erreur lors du chargement');
        console.error('Erreur lors du chargement des replays:', err);
      } finally {
        setIsLoading(false);
      }
    }
    loadGames();
  }, [currentScreen]);

  // Charger le replay sélectionné
  useEffect(() => {
    async function loadReplay() {
      if (!selectedGameId) return;

      try {
        setIsLoading(true);
        const replayData = await getReplay(selectedGameId);
        setReplay(replayData);
        setCurrentTurn(0);
        setIsPlaying(false);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Erreur lors du chargement du replay');
        console.error('Erreur lors du chargement du replay:', err);
      } finally {
        setIsLoading(false);
      }
    }
    loadReplay();
  }, [selectedGameId]);

  // Auto-play
  useEffect(() => {
    if (!isPlaying || !replay) return;

    const maxTurn = replay.states.length - 1;
    if (currentTurn >= maxTurn) {
      setIsPlaying(false);
      return;
    }

    const interval = setInterval(() => {
      setCurrentTurn((prev) => {
        if (prev >= maxTurn) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 1000); // 1 seconde par tour

    return () => clearInterval(interval);
  }, [isPlaying, currentTurn, replay]);

  const handlePreviousTurn = useCallback(() => {
    if (currentTurn > 0) {
      setCurrentTurn(currentTurn - 1);
      setIsPlaying(false);
    }
  }, [currentTurn]);

  const handleNextTurn = useCallback(() => {
    if (replay && currentTurn < replay.states.length - 1) {
      setCurrentTurn(currentTurn + 1);
      setIsPlaying(false);
    }
  }, [currentTurn, replay]);

  const handlePlayPause = useCallback(() => {
    if (!replay) return;
    const maxTurn = replay.states.length - 1;
    if (currentTurn >= maxTurn) {
      setCurrentTurn(0);
    }
    setIsPlaying(!isPlaying);
  }, [isPlaying, currentTurn, replay]);

  const currentState: GameStateView | null =
    replay && replay.states[currentTurn] ? replay.states[currentTurn] : null;

  // Gestion de la navigation
  const handleNavigate = (screen: Screen, mode?: GameMode, config?: GameConfig) => {
    setCurrentScreen(screen);
    if (mode) {
      setSelectedMode(mode);
    }
    if (config) {
      console.log('Game config:', config);
      // TODO: Utiliser la config pour démarrer le jeu
    }
  };

  const startPerfectionGame = async (config: GameConfig) => {
    if (config.mode !== 'perfection') {
      console.warn('Mode non supporté pour le live:', config.mode);
      return;
    }
    setLiveStartError(null);
    setIsStartingLiveGame(true);
    try {
      const session = await createPerfectionGame({
        opponents: config.opponents,
        difficulty: config.difficulty,
      });
      setLiveSession(session);
      setLiveSelectedUnitId(null);
      setLiveRuntimeError(null);
      setCurrentScreen('liveGame');
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Impossible de démarrer la partie.';
      setLiveStartError(message);
    } finally {
      setIsStartingLiveGame(false);
    }
  };

  const withLiveRequest = async (request: () => Promise<LiveGameStateResponse>) => {
    if (!liveSession) return;
    setIsLiveBusy(true);
    try {
      const updated = await request();
      setLiveSession(updated);
      setLiveRuntimeError(null);
      // Vérifier que l'unité sélectionnée existe toujours
      if (liveSelectedUnitId !== null) {
        const stillExists = updated.state.units.some(
          (unit) => (unit.id ?? -1) === liveSelectedUnitId
        );
        if (!stillExists) {
          setLiveSelectedUnitId(null);
        }
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Action live impossible.';
      setLiveRuntimeError(message);
    } finally {
      setIsLiveBusy(false);
    }
  };

  const handleLiveAction = async (actionId: number) => {
    if (!liveSession) return;
    await withLiveRequest(() => sendLiveAction(liveSession.gameId, actionId));
  };

  const handleLiveEndTurn = async () => {
    if (!liveSession) return;
    await withLiveRequest(() => endLiveTurn(liveSession.gameId));
  };

  const handleLiveRefresh = async () => {
    if (!liveSession) return;
    await withLiveRequest(() => getLiveGameState(liveSession.gameId));
  };

  const handleExitLiveGame = () => {
    setLiveSession(null);
    setLiveSelectedUnitId(null);
    setLiveRuntimeError(null);
    setCurrentScreen('modeSelection');
  };

  // Rendu conditionnel selon l'écran
  if (currentScreen === 'mainMenu') {
    return <MainMenu onNavigate={handleNavigate} />;
  }

  if (currentScreen === 'modeSelection') {
    return <ModeSelectionMenu onNavigate={handleNavigate} />;
  }

  if (currentScreen === 'gameSetup' && selectedMode) {
    const startProps =
      selectedMode === 'perfection'
        ? {
            onStartGame: startPerfectionGame,
            isStarting: isStartingLiveGame,
            startError: liveStartError,
          }
        : {};
    return (
      <GameSetupMenu
        mode={selectedMode}
        onNavigate={handleNavigate}
        {...startProps}
      />
    );
  }

  if (currentScreen === 'liveGame' && liveSession) {
    return (
      <LiveGameView
        session={liveSession}
        onExit={handleExitLiveGame}
        onSendAction={handleLiveAction}
        onEndTurn={handleLiveEndTurn}
        onRefresh={handleLiveRefresh}
        isBusy={isLiveBusy}
        selectedUnitId={liveSelectedUnitId}
        onSelectUnit={setLiveSelectedUnitId}
        error={liveRuntimeError}
      />
    );
  }

  // Écran "game" - Replay viewer (logique existante)
  return (
    <div className="min-h-screen bg-gray-900">
      {/* Header */}
      <header className="bg-gray-800 text-white p-4 shadow-lg">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <h1 className="text-2xl font-bold">Polytopia Replay Viewer</h1>
          <button
            onClick={() => handleNavigate('mainMenu')}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
          >
            ← Menu principal
          </button>
        </div>
      </header>

      {/* Contenu principal */}
      <main className="max-w-7xl mx-auto">
        {error && (
          <div className="bg-red-600 text-white p-4 m-4 rounded-lg">
            <strong>Erreur:</strong> {error}
            <button
              onClick={() => setError(null)}
              className="ml-4 underline"
            >
              Fermer
            </button>
          </div>
        )}

        {!selectedGameId ? (
          <GameList
            games={games}
            selectedGameId={selectedGameId}
            onSelectGame={setSelectedGameId}
            isLoading={isLoading}
          />
        ) : (
          <>
            {/* Bouton retour à la liste */}
            <div className="p-4">
              <button
                onClick={() => setSelectedGameId(null)}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
              >
                ← Retour à la liste
              </button>
            </div>

            {/* HUD */}
            {currentState && replay && (
              <HUD
                state={currentState}
                metadata={replay.metadata}
                currentTurn={currentTurn}
                maxTurn={replay.states.length - 1}
                onPreviousTurn={handlePreviousTurn}
                onNextTurn={handleNextTurn}
                onPlayPause={handlePlayPause}
                isPlaying={isPlaying}
              />
            )}

            {/* Plateau */}
            {currentState ? (
              <Board state={currentState} />
            ) : isLoading ? (
              <div className="p-8 text-center text-gray-400">
                Chargement...
              </div>
            ) : (
              <div className="p-8 text-center text-gray-400">
                Aucun état disponible
              </div>
            )}
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-gray-400 text-center p-4 mt-8">
        <p>Polytopia-JAX Replay Viewer</p>
      </footer>
    </div>
  );
}

export default App;

