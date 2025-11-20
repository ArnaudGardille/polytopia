import { useMemo } from 'react';
import type { LiveGameStateResponse } from '../types';
import { TerrainType } from '../types';
import { Board, type MoveTarget } from './Board';
import {
  Direction,
  encodeAttack,
  encodeMove,
  directionFromDelta,
} from '../utils/actionEncoder';

interface LiveGameViewProps {
  session: LiveGameStateResponse;
  onExit: () => void;
  onSendAction: (actionId: number) => void;
  onEndTurn: () => void;
  onRefresh: () => void;
  isBusy: boolean;
  selectedUnitId: number | null;
  onSelectUnit: (unitId: number | null) => void;
  error?: string | null;
}

const HUMAN_PLAYER_ID = 0;

type HexDirKey = 'NW' | 'NE' | 'W' | 'E' | 'SW' | 'SE';

const HEX_OFFSETS: Record<'even' | 'odd', Record<HexDirKey, [number, number]>> =
  {
    even: {
      NW: [-1, -1],
      NE: [0, -1],
      W: [-1, 0],
      E: [1, 0],
      SW: [-1, 1],
      SE: [0, 1],
    },
    odd: {
      NW: [0, -1],
      NE: [1, -1],
      W: [-1, 0],
      E: [1, 0],
      SW: [0, 1],
      SE: [1, 1],
    },
  };

const BUTTON_GRID: Array<Array<string>> = [
  ['NW', 'UP', 'NE'],
  ['W', 'CENTER', 'E'],
  ['SW', 'DOWN', 'SE'],
];

export function LiveGameView({
  session,
  onExit,
  onSendAction,
  onEndTurn,
  onRefresh,
  isBusy,
  selectedUnitId,
  onSelectUnit,
  error,
}: LiveGameViewProps) {
  const state = session.state;
  const boardHeight = state.terrain.length;
  const boardWidth = state.terrain[0]?.length || 0;
  const selectedUnit = useMemo(() => {
    if (selectedUnitId === null) return null;
    return state.units.find((unit) => (unit.id ?? -1) === selectedUnitId) || null;
  }, [selectedUnitId, state.units]);

  const getNeighborOffset = (
    dir: HexDirKey,
    row: number
  ): [number, number] | null => {
    const parity = row % 2 === 0 ? 'even' : 'odd';
    return HEX_OFFSETS[parity][dir] ?? null;
  };

  const getUnitAt = (x: number, y: number) =>
    state.units.find((unit) => unit.pos[0] === x && unit.pos[1] === y);

  const isWithinBounds = (x: number, y: number) =>
    x >= 0 && x < boardWidth && y >= 0 && y < boardHeight;

  const isTileTraversable = (x: number, y: number) => {
    if (!isWithinBounds(x, y)) return false;
    const terrain = state.terrain[y]?.[x];
    return (
      terrain === TerrainType.PLAIN ||
      terrain === TerrainType.FOREST
    );
  };

  const resolveDirKey = (button: string): HexDirKey | null => {
    if (
      button === 'NW' ||
      button === 'NE' ||
      button === 'W' ||
      button === 'E' ||
      button === 'SW' ||
      button === 'SE'
    ) {
      return button as HexDirKey;
    }
    return null;
  };

  const getTargetForDir = (dir: HexDirKey): [number, number] | null => {
    if (!selectedUnit) return null;
    const offset = getNeighborOffset(dir, selectedUnit.pos[1]);
    if (!offset) return null;
    const target: [number, number] = [
      selectedUnit.pos[0] + offset[0],
      selectedUnit.pos[1] + offset[1],
    ];
    if (
      target[0] < 0 ||
      target[0] >= boardWidth ||
      target[1] < 0 ||
      target[1] >= boardHeight
    ) {
      return null;
    }
    return target;
  };

  const getVerticalTarget = (
    direction: 'UP' | 'DOWN'
  ): [number, number] | null => {
    if (!selectedUnit) return null;
    const deltaY = direction === 'UP' ? -2 : 2;
    const target: [number, number] = [
      selectedUnit.pos[0],
      selectedUnit.pos[1] + deltaY,
    ];
    if (
      target[0] < 0 ||
      target[0] >= boardWidth ||
      target[1] < 0 ||
      target[1] >= boardHeight
    ) {
      return null;
    }
    return target;
  };

  const getDirectionFromOffset = (
    dx: number,
    dy: number,
    fromY: number
  ): Direction | null => {
    if (dx === 0 && dy === -2) return Direction.UP;
    if (dx === 0 && dy === 2) return Direction.DOWN;
    const parity = fromY % 2 === 0 ? 'even' : 'odd';
    const hexOffsets = HEX_OFFSETS[parity];
    for (const dir of Object.keys(hexOffsets) as HexDirKey[]) {
      const [offsetX, offsetY] = hexOffsets[dir];
      if (offsetX === dx && offsetY === dy) {
        return directionFromDelta([offsetX, offsetY]);
      }
    }
    return null;
  };

  const executeActionAtTarget = (target: [number, number]) => {
    if (!selectedUnit || selectedUnitId === null) return;
    const [targetX, targetY] = target;
    if (!isWithinBounds(targetX, targetY)) return;
    const occupant = getUnitAt(targetX, targetY);
    const unitId = selectedUnit.id ?? selectedUnitId ?? 0;
    if (occupant && occupant.owner === HUMAN_PLAYER_ID) {
      return;
    }
    if (occupant && occupant.owner !== HUMAN_PLAYER_ID) {
      const actionId = encodeAttack(unitId, target);
      onSendAction(actionId);
      return;
    }
    if (!isTileTraversable(targetX, targetY)) {
      return;
    }
    const dx = targetX - selectedUnit.pos[0];
    const dy = targetY - selectedUnit.pos[1];
    const direction = getDirectionFromOffset(dx, dy, selectedUnit.pos[1]);
    if (direction === null) return;
    const actionId = encodeMove(unitId, direction);
    onSendAction(actionId);
  };

  const handleHexCommand = (dir: HexDirKey) => {
    const target = getTargetForDir(dir);
    if (!target) return;
    executeActionAtTarget(target);
  };

  const handleVerticalCommand = (direction: 'UP' | 'DOWN') => {
    const target = getVerticalTarget(direction);
    if (!target) return;
    executeActionAtTarget(target);
  };

  const handleMoveToCell = (target: MoveTarget) => {
    executeActionAtTarget([target.x, target.y]);
  };

  const moveTargets = useMemo<MoveTarget[]>(() => {
    if (!selectedUnit) return [];

    const targets: MoveTarget[] = [];
    const collectTarget = (target: [number, number] | null) => {
      if (!target) return;
      const [targetX, targetY] = target;
      if (!isWithinBounds(targetX, targetY)) return;
      const occupant = getUnitAt(targetX, targetY);
      if (occupant && occupant.owner === HUMAN_PLAYER_ID) {
        return;
      }
      if (occupant && occupant.owner !== HUMAN_PLAYER_ID) {
        targets.push({ x: targetX, y: targetY, type: 'attack' });
        return;
      }
      if (!isTileTraversable(targetX, targetY)) {
        return;
      }
      const dx = targetX - selectedUnit.pos[0];
      const dy = targetY - selectedUnit.pos[1];
      const direction = getDirectionFromOffset(dx, dy, selectedUnit.pos[1]);
      if (direction !== null) {
        targets.push({ x: targetX, y: targetY, type: 'move' });
      }
    };

    (['NW', 'NE', 'W', 'E', 'SW', 'SE'] as HexDirKey[]).forEach((dir) =>
      collectTarget(getTargetForDir(dir))
    );
    (['UP', 'DOWN'] as const).forEach((direction) =>
      collectTarget(getVerticalTarget(direction))
    );
    return targets;
  }, [
    selectedUnit,
    boardHeight,
    boardWidth,
    state.units,
    state.terrain,
  ]);

  const turnProgress = Math.min(
    100,
    Math.round((state.turn / session.maxTurns) * 100)
  );

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col">
      <header className="bg-gray-800 p-4 shadow-lg flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-400">Mode Perfection Live</p>
          <h1 className="text-2xl font-bold">Partie #{session.gameId.slice(0, 8)}</h1>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={onRefresh}
            disabled={isBusy}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg disabled:opacity-50"
          >
            Rafraîchir
          </button>
          <button
            onClick={onExit}
            className="px-4 py-2 bg-red-600 hover:bg-red-500 rounded-lg"
          >
            Quitter
          </button>
        </div>
      </header>

      {error && (
        <div className="bg-red-600 text-white px-4 py-2 text-sm text-center">
          {error}
        </div>
      )}

      <main className="flex-1 grid grid-cols-1 lg:grid-cols-[2fr_1fr] gap-4 p-4">
        <div className="bg-gray-800/60 rounded-2xl p-4">
          <Board
            state={state}
            selectedUnitId={selectedUnitId}
            selectableUnitOwner={HUMAN_PLAYER_ID}
            onSelectUnit={(unitId, owner) => {
              if (owner === HUMAN_PLAYER_ID) {
                if (selectedUnitId === unitId) {
                  onSelectUnit(null);
                } else {
                  onSelectUnit(unitId);
                }
              }
            }}
            moveTargets={moveTargets}
            onSelectMoveTarget={handleMoveToCell}
          />
        </div>

        <aside className="bg-gray-800/60 rounded-2xl p-6 flex flex-col gap-6">
          <section>
            <h2 className="text-lg font-semibold mb-2">Progression</h2>
            <div className="text-sm text-gray-400 mb-1">
              Tour {state.turn} / {session.maxTurns}
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full"
                style={{ width: `${turnProgress}%` }}
              ></div>
            </div>
            {state.done && (
              <p className="mt-2 text-green-400 font-semibold">
                Partie terminée
              </p>
            )}
          </section>

          <section>
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-lg font-semibold">Unités</h2>
              <button
                onClick={() => onSelectUnit(null)}
                className="text-sm text-gray-400 hover:text-white"
              >
                Désélectionner
              </button>
            </div>
            <div className="max-h-40 overflow-y-auto pr-1">
              {state.units
                .filter((unit) => unit.owner === HUMAN_PLAYER_ID)
                .map((unit) => {
                  const unitId = unit.id ?? -1;
                  const isSelected = unitId === selectedUnitId;
                  return (
                    <button
                      key={unitId}
                      onClick={() => onSelectUnit(unitId)}
                      className={`w-full text-left px-3 py-2 rounded-lg mb-2 transition ${
                        isSelected
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-700 text-gray-200 hover:bg-gray-600'
                      }`}
                    >
                      Guerrier #{unitId} – ({unit.pos[0]}, {unit.pos[1]}) · HP{' '}
                      {unit.hp}
                    </button>
                  );
                })}
              {state.units.filter((u) => u.owner === HUMAN_PLAYER_ID).length ===
                0 && (
                <p className="text-sm text-gray-400">
                  Aucune unité disponible. Terminez votre tour.
                </p>
              )}
            </div>
          </section>

          <section>
            <h3 className="font-semibold mb-2">Déplacements / Attaques</h3>
            <div className="grid grid-cols-3 gap-2 max-w-xs">
              {BUTTON_GRID.flatMap((rowButtons, rowIndex) =>
                rowButtons.map((button, idx) => {
                  if (button === 'CENTER') {
                    return (
                      <div
                        key={`center-${rowIndex}-${idx}`}
                        className="flex items-center justify-center text-gray-600"
                      >
                        •
                      </div>
                    );
                  }
                  const isVerticalButton = button === 'UP' || button === 'DOWN';
                  const resolvedKey = !isVerticalButton
                    ? resolveDirKey(button)
                    : null;
                  if (!isVerticalButton && !resolvedKey) {
                    return (
                      <div
                        key={`placeholder-${rowIndex}-${idx}`}
                        className="flex items-center justify-center text-gray-600"
                      >
                        {button}
                      </div>
                    );
                  }
                  const target = isVerticalButton
                    ? getVerticalTarget(button as 'UP' | 'DOWN')
                    : resolvedKey
                    ? getTargetForDir(resolvedKey)
                    : null;
                  const occupant = target
                    ? state.units.find(
                        (unit) =>
                          unit.pos[0] === target[0] && unit.pos[1] === target[1]
                      )
                    : undefined;
                  const isFriendlyOccupant =
                    !!occupant && occupant.owner === HUMAN_PLAYER_ID;
                  const disabled =
                    !selectedUnit || !target || isBusy || isFriendlyOccupant;
                  const label =
                    button === 'UP'
                      ? '↑'
                      : button === 'DOWN'
                      ? '↓'
                      : button === 'W'
                      ? '←'
                      : button === 'E'
                      ? '→'
                      : button === 'NW'
                      ? '↖'
                      : button === 'NE'
                      ? '↗'
                      : button === 'SW'
                      ? '↙'
                      : '↘';
                  const handleClick = () => {
                    if (disabled || !target) return;
                    if (isVerticalButton) {
                      handleVerticalCommand(button as 'UP' | 'DOWN');
                    } else if (resolvedKey) {
                      handleHexCommand(resolvedKey);
                    }
                  };
                  return (
                    <button
                      key={`${button}-${rowIndex}-${idx}`}
                      onClick={handleClick}
                      disabled={disabled}
                      className="px-3 py-2 bg-blue-500/80 hover:bg-blue-500 rounded-lg disabled:opacity-40"
                      title={
                        target
                          ? `(${target[0]}, ${target[1]})${
                              occupant && occupant.owner !== HUMAN_PLAYER_ID
                                ? ' · Attaque'
                                : ''
                            }`
                          : undefined
                      }
                    >
                      {label}
                    </button>
                  );
                })
              )}
            </div>
          </section>

          <section className="flex flex-col gap-3">
            <button
              onClick={onEndTurn}
              disabled={isBusy}
              className="py-3 bg-purple-500 hover:bg-purple-600 rounded-xl font-semibold disabled:opacity-50"
            >
              Fin de tour
            </button>
          </section>
        </aside>
      </main>
    </div>
  );
}

