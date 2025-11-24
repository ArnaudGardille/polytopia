import { useMemo, useState } from 'react';
import type { LiveGameStateResponse, CityView } from '../types';
import { TerrainType, UnitType, ResourceType } from '../types';
import { Board, type MoveTarget } from './Board';
import { Scoreboard } from './Scoreboard';
import {
  Direction,
  directionFromDelta,
  encodeAttack,
  encodeMove,
  encodeResearchTech,
  encodeTrainUnit,
  encodeHarvestResource,
  getDirectionDelta,
} from '../utils/actionEncoder';
import { TECHNOLOGY_TREE } from '../data/techTree';
import { getResourceDefinition, HARVEST_ZONE_OFFSETS } from '../data/resources';

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

const TRAINABLE_UNITS = [
  {
    type: UnitType.WARRIOR,
    label: 'Guerrier',
    cost: 2,
  },
] as const;

const CITY_LEVEL_POP_THRESHOLDS = [0, 1, 3, 5] as const;
const CITY_STAR_INCOME_PER_LEVEL = [0, 2, 4, 6] as const;

type CityPopulationInfo = {
  value: number;
  hasData: boolean;
  next: number;
  progressPercent: number;
  isMaxLevel: boolean;
};

// Système de grille simple : 8 directions avec deltas {-1, 0, 1} en x et y
const NEIGHBOR_OFFSETS: [number, number][] = [
  [-1, -1], // UP_LEFT
  [0, -1],  // UP
  [1, -1],  // UP_RIGHT
  [1, 0],   // RIGHT
  [1, 1],   // DOWN_RIGHT
  [0, 1],   // DOWN
  [-1, 1],  // DOWN_LEFT
  [-1, 0],  // LEFT
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
  const [selectedCityPos, setSelectedCityPos] = useState<[number, number] | null>(
    null
  );
  const boardHeight = state.terrain.length;
  const boardWidth = state.terrain[0]?.length || 0;
  const playerStars = state.player_stars?.[HUMAN_PLAYER_ID] ?? 0;
  const isPlayerTurn = state.current_player === HUMAN_PLAYER_ID;
  const playerCities = useMemo(
    () => state.cities.filter((city) => city.owner === HUMAN_PLAYER_ID),
    [state.cities]
  );
  const playerUnits = useMemo(
    () => state.units.filter((unit) => unit.owner === HUMAN_PLAYER_ID),
    [state.units]
  );
  const playerTechRow = state.player_techs?.[HUMAN_PLAYER_ID] ?? [];
  const hasTech = (techId: number) =>
    techId >= 0 && Boolean(playerTechRow[techId]);
  const getTechName = (techId: number) =>
    TECHNOLOGY_TREE.find((tech) => tech.id === techId)?.name ?? `Tech ${techId}`;
  const unlockedTechCount = TECHNOLOGY_TREE.reduce(
    (count, tech) => count + (hasTech(tech.id) ? 1 : 0),
    0
  );
  const hasTechData = playerTechRow.length > 0;

  const getUnitAt = (x: number, y: number) =>
    state.units.find((unit) => unit.pos[0] === x && unit.pos[1] === y);

  const isCityOccupied = (city: CityView) =>
    Boolean(getUnitAt(city.pos[0], city.pos[1]));

  const selectedCity = useMemo(() => {
    if (!selectedCityPos) return null;
    return (
      state.cities.find(
        (city) =>
          city.pos[0] === selectedCityPos[0] && city.pos[1] === selectedCityPos[1]
      ) ?? null
    );
  }, [selectedCityPos, state.cities]);
  const selectedCityPopulation =
    selectedCity && state.city_population?.[selectedCity.pos[1]]?.[selectedCity.pos[0]];
  const cityPopulationInfo = useMemo<CityPopulationInfo | null>(() => {
    if (!selectedCity) return null;
    const maxLevelIndex = CITY_LEVEL_POP_THRESHOLDS.length - 1;
    const levelIndex = Math.min(
      Math.max(selectedCity.level, 0),
      maxLevelIndex
    );
    const nextLevelIndex = Math.min(levelIndex + 1, maxLevelIndex);
    const minThreshold = CITY_LEVEL_POP_THRESHOLDS[levelIndex];
    const nextThreshold = CITY_LEVEL_POP_THRESHOLDS[nextLevelIndex];
    const isMaxLevel = levelIndex >= maxLevelIndex;
    const baseValue =
      typeof selectedCityPopulation === 'number'
        ? selectedCityPopulation
        : minThreshold;
    const clampedValue = Math.max(
      minThreshold,
      Math.min(baseValue, nextThreshold)
    );
    const range = isMaxLevel ? 1 : Math.max(nextThreshold - minThreshold, 1);
    const filled = clampedValue - minThreshold;
    const progressPercent = isMaxLevel
      ? 100
      : Math.round((filled / range) * 100);
    return {
      value: baseValue,
      hasData: typeof selectedCityPopulation === 'number',
      next: nextThreshold,
      progressPercent,
      isMaxLevel,
    };
  }, [selectedCity, selectedCityPopulation]);
  const cityIncome = selectedCity
    ? CITY_STAR_INCOME_PER_LEVEL[
        Math.min(selectedCity.level, CITY_STAR_INCOME_PER_LEVEL.length - 1)
      ]
    : 0;
  const cityOccupied = selectedCity ? isCityOccupied(selectedCity) : false;
  const selectedUnit = useMemo(() => {
    if (selectedUnitId === null) return null;
    return state.units.find((unit) => (unit.id ?? -1) === selectedUnitId) || null;
  }, [selectedUnitId, state.units]);


  const handleSelectCity = (city: CityView) => {
    const nextPos: [number, number] = [city.pos[0], city.pos[1]];
    if (
      selectedCityPos &&
      selectedCityPos[0] === nextPos[0] &&
      selectedCityPos[1] === nextPos[1]
    ) {
      setSelectedCityPos(null);
      return;
    }
    setSelectedCityPos(nextPos);
    onSelectUnit(null);
  };

  const handleDeselectCity = () => {
    setSelectedCityPos(null);
  };

  const handleTrainUnit = (unitType: number) => {
    if (!selectedCity || !isPlayerTurn || isBusy) return;
    const actionId = encodeTrainUnit(unitType, selectedCity.pos);
    onSendAction(actionId);
  };

  const handleHarvestResource = (targetX: number, targetY: number) => {
    if (!selectedCity || !isPlayerTurn || isBusy) return;
    const actionId = encodeHarvestResource([targetX, targetY]);
    onSendAction(actionId);
  };

  const handleResearchTech = (techId: number) => {
    if (!isPlayerTurn || isBusy) return;
    const actionId = encodeResearchTech(techId);
    onSendAction(actionId);
  };

  const handleDeselectUnit = () => {
    onSelectUnit(null);
    setSelectedCityPos(null);
  };

  const handleUnitListClick = (unitId: number) => {
    if (selectedUnitId === unitId) {
      handleDeselectUnit();
    } else {
      setSelectedCityPos(null);
      onSelectUnit(unitId);
    }
  };

  const isWithinBounds = (x: number, y: number) =>
    x >= 0 && x < boardWidth && y >= 0 && y < boardHeight;

  const isTileTraversable = (x: number, y: number) => {
    if (!isWithinBounds(x, y)) return false;
    const terrain = state.terrain[y]?.[x];
    return (
      terrain === TerrainType.PLAIN ||
      terrain === TerrainType.PLAIN_FRUIT ||
      terrain === TerrainType.FOREST ||
      terrain === TerrainType.FOREST_WITH_WILD_ANIMAL
    );
  };

  const getDirectionFromOffset = (
    dx: number,
    dy: number
  ): Direction | null => {
    // Utiliser directement directionFromDelta pour convertir le delta en direction
    // Système simple : {-1, 0, 1} en x et {-1, 0, 1} en y
    return directionFromDelta([dx, dy]);
  };

  const executeActionAtTarget = (target: [number, number], validTargets?: MoveTarget[]) => {
    if (!selectedUnit || selectedUnitId === null) return;
    const [targetX, targetY] = target;
    if (!isWithinBounds(targetX, targetY)) return;
    
    // Vérifier que la case cible est dans la liste des cibles valides si fournie
    if (validTargets && !validTargets.some(t => t.x === targetX && t.y === targetY)) {
      console.warn(`Déplacement invalide : la case [${targetX}, ${targetY}] n'est pas dans la liste des cibles valides`);
      return;
    }
    
    const occupant = getUnitAt(targetX, targetY);
    const unitId = selectedUnit.id ?? selectedUnitId ?? 0;
    if (occupant && occupant.owner === HUMAN_PLAYER_ID) {
      return;
    }
    if (occupant && occupant.owner !== HUMAN_PLAYER_ID) {
      const actionId = encodeAttack(unitId, target);
      onSendAction(actionId);
      onSelectUnit(null);
      return;
    }
    if (!isTileTraversable(targetX, targetY)) {
      return;
    }
    const dx = targetX - selectedUnit.pos[0];
    const dy = targetY - selectedUnit.pos[1];
    const direction = getDirectionFromOffset(dx, dy);
    if (direction === null) {
      // Déplacement invalide : la case n'est pas adjacente ou la direction n'est pas reconnue
      console.warn(`Déplacement invalide depuis [${selectedUnit.pos[0]}, ${selectedUnit.pos[1]}] vers [${targetX}, ${targetY}] (dx=${dx}, dy=${dy})`);
      return;
    }
    const actionId = encodeMove(unitId, direction);
    onSendAction(actionId);
    onSelectUnit(null);
  };

  const handleMoveToCell = (target: MoveTarget) => {
    executeActionAtTarget([target.x, target.y], moveTargets);
  };

  const moveTargets = useMemo<MoveTarget[]>(() => {
    if (!selectedUnit) return [];
    if (selectedUnit.has_acted) return [];

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
      const direction = getDirectionFromOffset(dx, dy);
      if (direction !== null) {
        targets.push({ x: targetX, y: targetY, type: 'move' });
      }
    };

    // Générer les 8 voisins avec les deltas {-1, 0, 1}
    NEIGHBOR_OFFSETS.forEach(([dx, dy]) => {
      const targetX = selectedUnit.pos[0] + dx;
      const targetY = selectedUnit.pos[1] + dy;
      collectTarget([targetX, targetY]);
    });
    return targets;
  }, [
    selectedUnit,
    boardHeight,
    boardWidth,
    state.units,
    state.terrain,
  ]);

  const harvestableResources = useMemo(() => {
    if (!selectedCity) return [];
    const [cityX, cityY] = selectedCity.pos;
    const resourceTypes = state.resource_type ?? [];
    const resourceAvailability = state.resource_available ?? [];
    const list: {
      x: number;
      y: number;
      label: string;
      population: number;
      cost: number;
      available: boolean;
      missingTech: boolean;
      canHarvest: boolean;
    }[] = [];
    for (const [dx, dy] of HARVEST_ZONE_OFFSETS) {
      const targetX = cityX + dx;
      const targetY = cityY + dy;
      if (
        targetX < 0 ||
        targetX >= boardWidth ||
        targetY < 0 ||
        targetY >= boardHeight
      ) {
        continue;
      }
      const resourceType =
        resourceTypes?.[targetY]?.[targetX] ?? ResourceType.NONE;
      const def = getResourceDefinition(resourceType);
      if (!def || def.id === ResourceType.NONE) continue;
      const available = Boolean(resourceAvailability?.[targetY]?.[targetX]);
      const hasTechForResource =
        def.requiredTech === undefined
          ? true
          : Boolean(playerTechRow[def.requiredTech] ?? false);
      const canAfford = playerStars >= def.cost;
      list.push({
        x: targetX,
        y: targetY,
        label: def.name,
        population: def.population,
        cost: def.cost,
        available,
        missingTech: !hasTechForResource,
        canHarvest: available && hasTechForResource && canAfford,
      });
    }
    return list;
  }, [
    selectedCity,
    state.resource_type,
    state.resource_available,
    boardWidth,
    boardHeight,
    playerStars,
    playerTechRow,
  ]);

  const turnProgress = Math.min(
    100,
    Math.round((state.turn / session.maxTurns) * 100)
  );
  const humanScore = state.player_score?.[HUMAN_PLAYER_ID];
  const humanStars = state.player_stars?.[HUMAN_PLAYER_ID];
  const humanIncome = state.player_income?.[HUMAN_PLAYER_ID];

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

      <main className="flex-1 grid grid-cols-1 lg:grid-cols-[2fr_0.6fr] gap-4 p-4">
        <div className="bg-gray-800/60 rounded-2xl p-4">
          <Board
            state={state}
            selectedUnitId={selectedUnitId}
            selectableUnitOwner={HUMAN_PLAYER_ID}
            onDeselectUnit={handleDeselectUnit}
            selectedCityPos={selectedCityPos}
            selectableCityOwner={HUMAN_PLAYER_ID}
            onSelectCity={handleSelectCity}
            onDeselectCity={handleDeselectCity}
            onSelectUnit={(unitId, owner) => {
              if (owner === HUMAN_PLAYER_ID) {
                setSelectedCityPos(null);
                if (selectedUnitId === unitId) {
                  handleDeselectUnit();
                } else {
                  onSelectUnit(unitId);
                }
              }
            }}
            moveTargets={moveTargets}
            onSelectMoveTarget={handleMoveToCell}
          />
        </div>

        <aside className="bg-gray-800/60 rounded-2xl p-6 flex flex-col gap-6 lg:max-w-sm">
          <section>
            <h2 className="text-lg font-semibold mb-3">Statistiques</h2>
            <div className="grid grid-cols-3 gap-3 text-center">
              <div className="bg-gray-900/30 rounded-xl p-3">
                <p className="text-sm text-gray-400">Score</p>
                <p className="text-2xl font-bold">{humanScore ?? '—'}</p>
              </div>
              <div className="bg-gray-900/30 rounded-xl p-3">
                <p className="text-sm text-gray-400">Étoiles</p>
                <p className="text-2xl font-bold">
                  {humanStars !== undefined ? `${humanStars} ★` : '—'}
                </p>
              </div>
              <div className="bg-gray-900/30 rounded-xl p-3">
                <p className="text-sm text-gray-400">Gain</p>
                <p className="text-xl font-bold">
                  {humanIncome !== undefined ? `${humanIncome} ★/tour` : '—'}
                </p>
              </div>
            </div>
          </section>

          <section className="mt-4">
            <Scoreboard
              state={state}
              highlightPlayer={HUMAN_PLAYER_ID}
              title="Classement général"
              className="w-full"
            />
          </section>

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
                onClick={handleDeselectUnit}
                className="text-sm text-gray-400 hover:text-white"
              >
                Désélectionner
              </button>
            </div>
            <div className="max-h-40 overflow-y-auto pr-1">
              {playerUnits.map((unit) => {
                const unitId = unit.id ?? -1;
                const isSelected = unitId === selectedUnitId;
                return (
                  <button
                    key={unitId}
                    onClick={() => handleUnitListClick(unitId)}
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
              {playerUnits.length === 0 && (
                <p className="text-sm text-gray-400">
                  Aucune unité disponible. Terminez votre tour.
                </p>
              )}
            </div>
          </section>

          <section>
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-lg font-semibold">Villes</h2>
              <span className="text-sm text-amber-300 font-semibold">
                ★ {playerStars}
              </span>
            </div>
            <div className="max-h-32 overflow-y-auto pr-1">
              {playerCities.map((city, idx) => {
                const isSelected =
                  selectedCityPos?.[0] === city.pos[0] &&
                  selectedCityPos?.[1] === city.pos[1];
                return (
                  <button
                    key={`${city.pos[0]}-${city.pos[1]}-${idx}`}
                    onClick={() => handleSelectCity(city)}
                    className={`w-full text-left px-3 py-2 rounded-lg mb-2 transition ${
                      isSelected
                        ? 'bg-cyan-600 text-white'
                        : 'bg-gray-700 text-gray-200 hover:bg-gray-600'
                    }`}
                  >
                    Ville ({city.pos[0]}, {city.pos[1]}) · Niveau {city.level}
                  </button>
                );
              })}
              {playerCities.length === 0 && (
                <p className="text-sm text-gray-400">
                  Aucune ville sous votre contrôle.
                </p>
              )}
            </div>
            {selectedCity && (
              <div className="mt-3 space-y-3">
                <div className="flex items-center justify-between text-sm text-gray-300">
                  <span>
                    Ville ({selectedCity.pos[0]}, {selectedCity.pos[1]}) · Niveau{' '}
                    {selectedCity.level}
                  </span>
                  <button
                    onClick={handleDeselectCity}
                    className="text-xs text-gray-400 hover:text-white"
                  >
                    Fermer
                  </button>
                </div>
                <div className="flex flex-wrap gap-2">
                  {TRAINABLE_UNITS.map((unit) => {
                    const affordable = playerStars >= unit.cost;
                    const disabled =
                      !isPlayerTurn || isBusy || !affordable || cityOccupied;
                    const title = !isPlayerTurn
                      ? 'Attendez votre tour'
                      : cityOccupied
                      ? 'Case occupée'
                      : !affordable
                      ? "Pas assez d'étoiles"
                      : undefined;
                    return (
                      <button
                        key={unit.type}
                        onClick={() => handleTrainUnit(unit.type)}
                        disabled={disabled}
                        className="px-3 py-2 bg-emerald-600 hover:bg-emerald-500 rounded-lg text-sm font-semibold disabled:opacity-40"
                        title={title}
                      >
                        {unit.label} ({unit.cost}★)
                      </button>
                    );
                  })}
                </div>
                {!isPlayerTurn && (
                  <p className="text-xs text-orange-300">
                    Patientez jusqu'à votre tour pour entraîner une unité.
                  </p>
                )}
                {cityOccupied && (
                  <p className="text-xs text-orange-300">
                    La case est occupée, libérez-la avant de recruter.
                  </p>
                )}
                <div className="bg-gray-900/30 rounded-lg p-3 space-y-2">
                  <div className="flex items-center justify-between">
                    <p className="text-sm text-gray-200">
                      Niveau {selectedCity.level}
                    </p>
                    <p className="text-sm font-semibold text-amber-300">
                      {cityIncome}★/tour
                    </p>
                  </div>
                  <p className="text-xs text-gray-400">
                    Les villes ne stockent pas d'étoiles : elles alimentent ton revenu
                    de fin de tour via leur niveau et leur évolution.
                  </p>
                  {cityPopulationInfo && (
                    <>
                      <div className="flex items-center justify-between text-xs text-gray-400">
                        <span>Population</span>
                        <span>
                          {cityPopulationInfo.hasData
                            ? `${cityPopulationInfo.value}/${cityPopulationInfo.next}`
                            : `— / ${cityPopulationInfo.next}`}
                        </span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div
                          className="h-2 rounded-full bg-emerald-400 transition-all"
                          style={{ width: `${cityPopulationInfo.progressPercent}%` }}
                        />
                      </div>
                      <p className="text-xs text-gray-400">
                        {cityPopulationInfo.isMaxLevel
                          ? 'Niveau max atteint'
                          : `Prochain niveau à ${cityPopulationInfo.next} pop`}
                      </p>
                    </>
                  )}
                </div>
                <div className="bg-gray-900/30 rounded-lg p-3 space-y-2">
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-semibold">
                      Ressources adjacentes
                    </h4>
                    <span className="text-xs text-gray-400">
                      Zone de récolte
                    </span>
                  </div>
                  {harvestableResources.length === 0 ? (
                    <p className="text-xs text-gray-400">
                      Aucune ressource exploitable.
                    </p>
                  ) : (
                    <div className="space-y-2">
                      {harvestableResources.map((resource) => {
                        const disabled =
                          !resource.canHarvest || !isPlayerTurn || isBusy;
                        const title = !resource.available
                          ? 'Ressource déjà récoltée'
                          : resource.missingTech
                          ? 'Technologie requise'
                          : playerStars < resource.cost
                          ? "Pas assez d'étoiles"
                          : undefined;
                        return (
                          <div
                            key={`${resource.x}-${resource.y}-${resource.label}`}
                            className="flex items-center justify-between gap-2 text-sm"
                          >
                            <div>
                              <p className="font-semibold">
                                {resource.label}{' '}
                                <span className="text-xs text-gray-400">
                                  ({resource.x}, {resource.y}) · +
                                  {resource.population} pop
                                </span>
                              </p>
                              {!resource.available && (
                                <p className="text-xs text-gray-500">
                                  Épuisée
                                </p>
                              )}
                              {resource.missingTech && (
                                <p className="text-xs text-amber-400">
                                  Tech requise
                                </p>
                              )}
                            </div>
                            <button
                              onClick={() =>
                                handleHarvestResource(resource.x, resource.y)
                              }
                              disabled={disabled}
                              className="px-3 py-1.5 bg-amber-500 hover:bg-amber-400 rounded-lg text-xs font-semibold disabled:opacity-40"
                              title={title}
                            >
                              Récolter ({resource.cost}★)
                            </button>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            )}
          </section>

          <section>
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-lg font-semibold">Technologies</h2>
              <span className="text-sm text-gray-400">
                {unlockedTechCount}/{TECHNOLOGY_TREE.length}
              </span>
            </div>
            {!hasTechData ? (
              <p className="text-sm text-gray-400">
                Arbre des technologies indisponible sur cette partie.
              </p>
            ) : (
              <div className="space-y-3 max-h-80 overflow-y-auto pr-1">
                {TECHNOLOGY_TREE.map((tech) => {
                  const unlocked = hasTech(tech.id);
                  const dependencyStates = tech.dependencies.map((dep) => ({
                    id: dep,
                    name: getTechName(dep),
                    fulfilled: hasTech(dep),
                  }));
                  const missingDeps = dependencyStates.filter(
                    (dep) => !dep.fulfilled
                  );
                  const depsMet = missingDeps.length === 0;
                  const affordable = playerStars >= tech.cost;
                  const available = !unlocked && depsMet && affordable;
                  const statusLabel = unlocked
                    ? 'Debloquee'
                    : !depsMet
                    ? 'Prerequis manquants'
                    : affordable
                    ? 'Disponible'
                    : 'Trop couteuse';
                  const statusColor = unlocked
                    ? 'text-emerald-300'
                    : available
                    ? 'text-cyan-300'
                    : 'text-orange-300';
                  const disabledReason = !isPlayerTurn
                    ? 'Attendez votre tour'
                    : isBusy
                    ? 'Action en cours'
                    : unlocked
                    ? 'Technologie deja debloquee'
                    : !depsMet
                    ? `Prerequis manquants: ${missingDeps
                        .map((dep) => dep.name)
                        .join(', ')}`
                    : !affordable
                    ? "Pas assez d'etoiles"
                    : undefined;
                  const canResearch = available && isPlayerTurn && !isBusy;
                  return (
                    <div
                      key={tech.id}
                      className={`p-4 rounded-2xl border transition ${
                        unlocked
                          ? 'border-emerald-500/40 bg-emerald-900/10'
                          : available
                          ? 'border-cyan-500/40 bg-cyan-900/10'
                          : 'border-gray-700 bg-gray-900/30'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div>
                          <p className="font-semibold text-base">{tech.name}</p>
                          <p className="text-xs text-gray-400 mt-1">
                            {tech.description}
                          </p>
                        </div>
                        <div className="text-right">
                          <div className="text-sm text-gray-200 font-semibold">
                            {tech.cost} ★
                          </div>
                          <div className={`text-xs font-semibold ${statusColor}`}>
                            {statusLabel}
                          </div>
                        </div>
                      </div>
                      <div className="mt-3">
                        <p className="text-xs font-semibold text-gray-300 mb-1">
                          Prerequis
                        </p>
                        {dependencyStates.length === 0 ? (
                          <p className="text-xs text-gray-400">Aucun</p>
                        ) : (
                          <div className="flex flex-wrap gap-2">
                            {dependencyStates.map((dep) => (
                              <span
                                key={dep.id}
                                className={`px-2 py-1 rounded-full text-xs ${
                                  dep.fulfilled
                                    ? 'bg-emerald-500/20 text-emerald-200'
                                    : 'bg-orange-500/10 text-orange-200'
                                }`}
                              >
                                {dep.name}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                      <div className="mt-3">
                        <p className="text-xs font-semibold text-gray-300 mb-1">
                          Effets
                        </p>
                        <ul className="list-disc text-xs text-gray-300 ml-4 space-y-1">
                          {tech.unlocks.map((unlock, idx) => (
                            <li key={idx}>{unlock}</li>
                          ))}
                        </ul>
                      </div>
                      <button
                        onClick={() => handleResearchTech(tech.id)}
                        disabled={!canResearch}
                        title={disabledReason}
                        className="mt-3 w-full py-2 rounded-xl font-semibold text-sm bg-purple-600 hover:bg-purple-500 disabled:opacity-40 disabled:cursor-not-allowed"
                      >
                        Rechercher ({tech.cost}★)
                      </button>
                      {!isPlayerTurn && (
                        <p className="text-xs text-orange-300 mt-1">
                          Patientez jusqua votre tour pour lancer une recherche.
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
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
