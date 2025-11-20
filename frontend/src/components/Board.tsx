import { useMemo, useEffect, useState } from 'react';
import type { GameStateView } from '../types';
import { getTerrainIcon, getPlayerColor, getUnitIcon, getCityIcon } from '../utils/iconMapper';

// Proportions tirées du sprite haute résolution (Grass.png)
const BASE_TILE_WIDTH = 2067;
const BASE_TILE_HEIGHT = 2249;
const TILE_HEIGHT_OVER_WIDTH = BASE_TILE_HEIGHT / BASE_TILE_WIDTH; // ≈ 1.088
// Ajustement empirique : seule ~68 % de la hauteur (la partie « losange ») doit rester visible
const TOP_OVERLAP_RATIO = 0.68;

interface BoardProps {
  state: GameStateView;
  cellSize?: number;
  selectedUnitId?: number | null;
  selectableUnitOwner?: number | null;
  onSelectUnit?: (unitId: number, owner: number) => void;
}

// Fonction pour calculer la position d'un hexagone en grille hexagonale (pointy-top)
function hexToPixel(x: number, y: number, tileWidth: number): [number, number] {
  const tileHeight = tileWidth * TILE_HEIGHT_OVER_WIDTH;
  const verticalPitch = tileHeight * (1 - TOP_OVERLAP_RATIO);
  const pixelX = tileWidth * (x + 0.5 * (y % 2));
  const pixelY = tileHeight / 2 + y * verticalPitch;
  return [pixelX, pixelY];
}

// Conversion offset odd-r vers coordonnées diagonales (cube axes x et y)
function offsetToDiagonalAxes(x: number, y: number): [number, number] {
  const cubeX = x - (y - (y & 1)) / 2;
  const cubeZ = y;
  const cubeY = -cubeX - cubeZ;
  return [cubeX, cubeY];
}

export function Board({
  state,
  cellSize: propCellSize,
  selectedUnitId,
  selectableUnitOwner,
  onSelectUnit,
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

  return (
    <div className="board-container">
      <svg
        className="board-svg"
        viewBox={viewBox}
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Grille de terrain avec hexagones */}
        {terrain.map((row, y) =>
          row.map((terrainType, x) => {
            const terrainIcon = getTerrainIcon(terrainType);
            const [centerX, centerY] = hexToPixel(x, y, tileWidth);
            const [diagX, diagY] = offsetToDiagonalAxes(x, y);
            const imageX = centerX - tileWidth / 2;
            const imageY = centerY - tileHeight / 2;
            
            return (
              <g key={`terrain-${x}-${y}`}>
                {/* Image de terrain si disponible */}
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
                      // Cacher l'image si elle ne charge pas
                      (e.target as SVGImageElement).style.display = 'none';
                    }}
                  />
                )}
                {/* Affichage des coordonnées pour debug */}
                <text
                  x={centerX}
                  y={centerY}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fill="white"
                  fontSize={Math.min(tileWidth, tileHeight) * 0.15}
                  fontWeight="bold"
                  stroke="black"
                  strokeWidth={Math.min(tileWidth, tileHeight) * 0.02}
                  paintOrder="stroke"
                >
                  {diagX},{diagY}
                </text>
              </g>
            );
          })
        )}

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
          
          return (
            <g key={`city-${idx}`} className="city-marker">
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
          const handleClick = () => {
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

