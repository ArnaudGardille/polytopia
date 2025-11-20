import { useMemo, useEffect, useState, type MouseEvent } from 'react';
import type { GameStateView, CityView } from '../types';
import { getTerrainIcon, getPlayerColor, getUnitIcon, getCityIcon } from '../utils/iconMapper';
import { HARVEST_ZONE_OFFSETS } from '../data/resources';

// Proportions tirées du sprite haute résolution (Grass.png)
const BASE_TILE_WIDTH = 2067;
const BASE_TILE_HEIGHT = 2249;
const TILE_HEIGHT_OVER_WIDTH = BASE_TILE_HEIGHT / BASE_TILE_WIDTH; // ≈ 1.088
// Ajustement empirique : seule ~68 % de la hauteur (la partie « losange ») doit rester visible
const TOP_OVERLAP_RATIO = 0.68;

export type MoveTarget = {
  x: number;
  y: number;
  type: 'move' | 'attack';
};

interface BoardProps {
  state: GameStateView;
  cellSize?: number;
  selectedUnitId?: number | null;
  selectableUnitOwner?: number | null;
  onSelectUnit?: (unitId: number, owner: number) => void;
  onDeselectUnit?: () => void;
  selectedCityPos?: [number, number] | null;
  selectableCityOwner?: number | null;
  onSelectCity?: (city: CityView) => void;
  onDeselectCity?: () => void;
  moveTargets?: MoveTarget[];
  onSelectMoveTarget?: (target: MoveTarget) => void;
}

// Fonction pour calculer la position d'un hexagone en grille hexagonale (pointy-top)
function hexToPixel(x: number, y: number, tileWidth: number): [number, number] {
  const tileHeight = tileWidth * TILE_HEIGHT_OVER_WIDTH;
  const verticalPitch = tileHeight * (1 - TOP_OVERLAP_RATIO);
  const pixelX = tileWidth * (x + 0.5 * (y % 2));
  const pixelY = tileHeight / 2 + y * verticalPitch;
  return [pixelX, pixelY];
}

export function Board({
  state,
  cellSize: propCellSize,
  selectedUnitId,
  selectableUnitOwner,
  onSelectUnit,
  onDeselectUnit,
  selectedCityPos,
  selectableCityOwner,
  onSelectCity,
  onDeselectCity,
  moveTargets,
  onSelectMoveTarget,
}: BoardProps) {
  const { terrain, cities, units } = state;
  const height = terrain.length;
  const width = terrain[0]?.length || 0;
  const [containerSize, setContainerSize] = useState({ width: 800, height: 600 });

  // Calculer la taille de cellule responsive (largeur d'un hexagone)
  const hexSize = useMemo(() => {
    if (propCellSize) return propCellSize;
    const maxWidth = Math.min(containerSize.width - 40, 1600);
    const maxHeight = containerSize.height - 40;
    const tileHeightUnit = TILE_HEIGHT_OVER_WIDTH;
    const verticalPitchUnit = tileHeightUnit * (1 - TOP_OVERLAP_RATIO);
    const widthDenominator = width > 0 ? width + 0.5 : 1;
    const heightDenominator = height > 0 ? tileHeightUnit + (height - 1) * verticalPitchUnit : tileHeightUnit;
    const cellSizeByWidth = maxWidth / widthDenominator;
    const cellSizeByHeight = maxHeight / heightDenominator;
    return Math.min(cellSizeByWidth, cellSizeByHeight, 160);
  }, [width, height, containerSize, propCellSize]);

  const tileWidth = hexSize;
  const tileHeight = tileWidth * TILE_HEIGHT_OVER_WIDTH;
  const verticalPitch = tileHeight * (1 - TOP_OVERLAP_RATIO);
  const spriteCenterCorrectionY = verticalPitch / 2;

  // Mettre à jour la taille du conteneur
  useEffect(() => {
    const updateSize = () => {
      const container = document.querySelector('.board-container');
      if (container) {
        setContainerSize({
          width: container.clientWidth,
          height: window.innerHeight - 300, // Réserver de l'espace pour le HUD
        });
      }
    };

    updateSize();
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, []);

  // Calculer les dimensions du viewBox pour la grille hexagonale (pointy-top)
  const viewBox = useMemo(() => {
    const paddingX = tileWidth * 0.05;
    const paddingY = tileHeight * 0.05;
    const boardWidth = width > 0 ? width * tileWidth + tileWidth / 2 : tileWidth;
    const boardHeight = height > 0 ? tileHeight + (height - 1) * verticalPitch : tileHeight;
    const minX = -tileWidth / 2 - paddingX;
    const minY = -paddingY;
    const viewWidth = boardWidth + paddingX * 2;
    const viewHeight = boardHeight + paddingY * 2;
    return `${minX} ${minY} ${viewWidth} ${viewHeight}`;
  }, [width, height, tileWidth, tileHeight, verticalPitch]);

  const moveTargetsMap = useMemo(() => {
    if (!moveTargets || moveTargets.length === 0) return null;
    const map = new Map<string, MoveTarget>();
    moveTargets.forEach((target) => {
      map.set(`${target.x}-${target.y}`, target);
    });
    return map;
  }, [moveTargets]);

  const handleBackgroundClick = () => {
    if (onDeselectUnit) {
      onDeselectUnit();
    }
    if (onDeselectCity) {
      onDeselectCity();
    }
  };

  const harvestZone = useMemo(() => {
    if (!selectedCityPos) return null;
    const [cityX, cityY] = selectedCityPos;
    const zone = new Set<string>();
    HARVEST_ZONE_OFFSETS.forEach(([dx, dy]) => {
      const tx = cityX + dx;
      const ty = cityY + dy;
      if (tx >= 0 && tx < width && ty >= 0 && ty < height) {
        zone.add(`${tx}-${ty}`);
      }
    });
    return zone;
  }, [selectedCityPos, width, height]);

  return (
    <div className="board-container" onClick={handleBackgroundClick}>
      <svg
        className="board-svg"
        viewBox={viewBox}
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Grille de terrain */}
        {terrain.map((row, y) =>
          row.map((terrainType, x) => {
            const terrainIcon = getTerrainIcon(terrainType);
            const [centerX, centerY] = hexToPixel(x, y, tileWidth);
            const imageX = centerX - tileWidth / 2;
            const imageY = centerY - tileHeight / 2;
            const inHarvestZone = harvestZone?.has(`${x}-${y}`) ?? false;
            return (
              <g key={`terrain-${x}-${y}`}>
                {terrainIcon && (
                  <image
                    href={terrainIcon}
                    x={imageX}
                    y={imageY}
                    width={tileWidth}
                    height={tileHeight}
                    preserveAspectRatio="xMidYMid meet"
                    opacity="0.9"
                    onError={(e) => {
                      (e.target as SVGImageElement).style.display = 'none';
                    }}
                  />
                )}
                {inHarvestZone && (
                  <circle
                    cx={centerX}
                    cy={centerY + spriteCenterCorrectionY}
                    r={Math.min(tileWidth, tileHeight) * 0.25}
                    fill="rgba(251, 191, 36, 0.25)"
                    stroke="#fbbf24"
                    strokeWidth={Math.max(1, tileWidth * 0.02)}
                    pointerEvents="none"
                  />
                )}
              </g>
            );
          })
        )}

        {/* Cibles de déplacement au premier plan */}
        {moveTargetsMap &&
          [...moveTargetsMap.values()].map((moveTarget) => {
            const [centerX, centerY] = hexToPixel(
              moveTarget.x,
              moveTarget.y,
              tileWidth
            );
            return (
              <g
                key={`movetarget-${moveTarget.x}-${moveTarget.y}`}
                onClick={(event) => {
                  event.stopPropagation();
                  if (onSelectMoveTarget) {
                    onSelectMoveTarget(moveTarget);
                  }
                }}
                style={{
                  cursor: onSelectMoveTarget ? 'pointer' : undefined,
                }}
              >
                <circle
                  cx={centerX}
                  cy={centerY}
                  r={Math.min(tileWidth, tileHeight) * 0.35}
                  fill={
                    moveTarget.type === 'attack'
                      ? 'rgba(239,68,68,0.2)'
                      : 'rgba(59,130,246,0.2)'
                  }
                />
                <circle
                  cx={centerX}
                  cy={centerY}
                  r={Math.min(tileWidth, tileHeight) * 0.28}
                  fill={
                    moveTarget.type === 'attack'
                      ? 'rgba(239,68,68,0.35)'
                      : 'rgba(59,130,246,0.35)'
                  }
                  stroke={
                    moveTarget.type === 'attack' ? '#ef4444' : '#3b82f6'
                  }
                  strokeWidth={Math.min(tileWidth, tileHeight) * 0.04}
                />
                <circle
                  cx={centerX}
                  cy={centerY}
                  r={Math.min(tileWidth, tileHeight) * 0.16}
                  fill={
                    moveTarget.type === 'attack'
                      ? 'rgba(239,68,68,0.75)'
                      : 'rgba(59,130,246,0.75)'
                  }
                />
              </g>
            );
          })}

        {/* Villes avec images */}
        {cities.map((city, idx) => {
          const [x, y] = city.pos;
          const [centerX, centerY] = hexToPixel(x, y, tileWidth);
          const playerColor = getPlayerColor(city.owner);
          const cityIcon = getCityIcon(city.level, city.owner);
          const iconSize = tileWidth * 0.65;
          const iconCenterY = centerY - spriteCenterCorrectionY;
          const iconX = centerX - iconSize / 2;
          const iconY = iconCenterY - iconSize / 2;
          const levelBadgeY = iconCenterY + iconSize * 0.2;
          const isSelected =
            selectedCityPos?.[0] === x && selectedCityPos?.[1] === y;
          const isSelectable =
            typeof selectableCityOwner === 'number'
              ? selectableCityOwner === city.owner
              : true;
          const handleCityClick = (event: MouseEvent<SVGGElement>) => {
            event.stopPropagation();
            if (onSelectCity && isSelectable) {
              onSelectCity(city);
            }
          };

          return (
            <g
              key={`city-${idx}`}
              className="city-marker"
              onClick={handleCityClick}
              style={{
                cursor: onSelectCity && isSelectable ? 'pointer' : undefined,
              }}
            >
              {isSelected && (
                <circle
                  cx={centerX}
                  cy={iconCenterY}
                  r={Math.min(tileWidth, tileHeight) * 0.38}
                  fill="none"
                  stroke="#38bdf8"
                  strokeWidth={Math.min(tileWidth, tileHeight) * 0.04}
                  strokeDasharray="6 4"
                />
              )}
              {cityIcon ? (
                <>
                  {/* Image de ville si disponible */}
                  <image
                    href={cityIcon}
                    x={iconX}
                    y={iconY}
                    width={iconSize}
                    height={iconSize}
                    preserveAspectRatio="xMidYMid meet"
                    opacity="0.95"
                    onError={(e) => {
                      (e.target as SVGImageElement).style.display = 'none';
                    }}
                  />
                  {/* Badge de niveau en bas à droite */}
                  <circle
                    cx={centerX + iconSize * 0.25}
                    cy={levelBadgeY}
                    r={Math.min(tileWidth, tileHeight) * 0.12}
                    fill={playerColor}
                    stroke="#fff"
                    strokeWidth="1.5"
                  />
                  <text
                    x={centerX + iconSize * 0.25}
                    y={levelBadgeY}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fill="white"
                    fontSize={Math.min(tileWidth, tileHeight) * 0.14}
                    fontWeight="bold"
                  >
                    {city.level}
                  </text>
                </>
              ) : (
                <>
                  {/* Fallback: cercle coloré si pas d'image */}
                  <circle
                    cx={centerX}
                    cy={iconCenterY}
                    r={Math.min(tileWidth, tileHeight) * 0.25}
                    fill={playerColor}
                    stroke="#fff"
                    strokeWidth="2"
                  />
                  <text
                    x={centerX}
                    y={iconCenterY}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fill="white"
                    fontSize={Math.min(tileWidth, tileHeight) * 0.3}
                    fontWeight="bold"
                  >
                    {city.level}
                  </text>
                </>
              )}
            </g>
          );
        })}

        {/* Unités avec images */}
        {units.map((unit, idx) => {
          const [x, y] = unit.pos;
          const [centerX, centerY] = hexToPixel(x, y, tileWidth);
          const playerColor = getPlayerColor(unit.owner);
          const unitIcon = getUnitIcon(unit.type, unit.owner);
          const iconSize = tileWidth * 0.55;
          const iconCenterY = centerY - spriteCenterCorrectionY;
          const iconX = centerX - iconSize / 2;
          const iconY = iconCenterY - iconSize / 2;
          const badgeY = iconCenterY + iconSize * 0.18;
          const unitId = unit.id ?? idx;
          const isSelected = selectedUnitId === unitId;
          const isSelectable =
            typeof selectableUnitOwner === 'number'
              ? selectableUnitOwner === unit.owner
              : true;
          const handleClick = (event: MouseEvent<SVGGElement>) => {
            event.stopPropagation();
            if (onSelectUnit && isSelectable) {
              onSelectUnit(unitId, unit.owner);
            }
          };
          const cursorStyle =
            onSelectUnit && isSelectable ? { cursor: 'pointer' } : undefined;
          
          return (
            <g
              key={`unit-${idx}`}
              className="unit-animation"
              onClick={handleClick}
              style={cursorStyle}
            >
              {isSelected && (
                <circle
                  cx={centerX}
                  cy={iconCenterY}
                  r={Math.min(tileWidth, tileHeight) * 0.35}
                  fill="none"
                  stroke="#facc15"
                  strokeWidth={Math.min(tileWidth, tileHeight) * 0.04}
                  strokeDasharray="4 4"
                />
              )}
              {/* Image de l'unité si disponible */}
              {unitIcon ? (
                <>
                  <image
                    href={unitIcon}
                    x={iconX}
                    y={iconY}
                    width={iconSize}
                    height={iconSize}
                    preserveAspectRatio="xMidYMid meet"
                    onError={(e) => {
                      // Fallback vers le cercle si l'image ne charge pas
                      (e.target as SVGImageElement).style.display = 'none';
                    }}
                  />
                  {/* Badge HP en bas à droite */}
                  <circle
                      cx={centerX + iconSize * 0.2}
                      cy={badgeY}
                    r={Math.min(tileWidth, tileHeight) * 0.1}
                    fill={playerColor}
                    stroke="#fff"
                    strokeWidth="1.5"
                  />
                  <text
                    x={centerX + iconSize * 0.2}
                    y={badgeY}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fill="white"
                    fontSize={Math.min(tileWidth, tileHeight) * 0.12}
                    fontWeight="bold"
                  >
                    {unit.hp}
                  </text>
                </>
              ) : (
                <>
                  {/* Fallback: cercle coloré si pas d'image */}
                  <circle
                    cx={centerX}
                    cy={iconCenterY}
                    r={Math.min(tileWidth, tileHeight) * 0.2}
                    fill={playerColor}
                    stroke="#fff"
                    strokeWidth="2"
                  />
                  <text
                    x={centerX}
                    y={iconCenterY + iconSize * 0.15}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fill="white"
                    fontSize={Math.min(tileWidth, tileHeight) * 0.2}
                    fontWeight="bold"
                  >
                    {unit.hp}
                  </text>
                </>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
