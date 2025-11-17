import { useMemo, useEffect, useState } from 'react';
import type { GameStateView } from '../types';
import { getTerrainColor, getTerrainIcon, getPlayerColor, getUnitIcon, getCityIcon } from '../utils/iconMapper';

interface BoardProps {
  state: GameStateView;
  cellSize?: number;
}

export function Board({ state, cellSize: propCellSize }: BoardProps) {
  const { terrain, cities, units } = state;
  const height = terrain.length;
  const width = terrain[0]?.length || 0;
  const [containerSize, setContainerSize] = useState({ width: 800, height: 600 });

  // Calculer la taille de cellule responsive
  const cellSize = useMemo(() => {
    if (propCellSize) return propCellSize;
    const maxWidth = Math.min(containerSize.width - 40, 1200);
    const maxHeight = containerSize.height - 40;
    const cellSizeByWidth = maxWidth / width;
    const cellSizeByHeight = maxHeight / height;
    return Math.min(cellSizeByWidth, cellSizeByHeight, 60);
  }, [width, height, containerSize, propCellSize]);

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

  const viewBox = useMemo(() => {
    return `0 0 ${width * cellSize} ${height * cellSize}`;
  }, [width, height, cellSize]);

  return (
    <div className="board-container">
      <svg
        className="board-svg"
        viewBox={viewBox}
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Grille de terrain avec images */}
        {terrain.map((row, y) =>
          row.map((terrainType, x) => {
            const terrainIcon = getTerrainIcon(terrainType);
            return (
              <g key={`terrain-${x}-${y}`}>
                {/* Fond de couleur en fallback */}
                <rect
                  x={x * cellSize}
                  y={y * cellSize}
                  width={cellSize}
                  height={cellSize}
                  fill={getTerrainColor(terrainType)}
                  stroke="#8B7355"
                  strokeWidth="1"
                />
                {/* Image de terrain si disponible */}
                {terrainIcon && (
                  <image
                    href={terrainIcon}
                    x={x * cellSize}
                    y={y * cellSize}
                    width={cellSize}
                    height={cellSize}
                    preserveAspectRatio="xMidYMid slice"
                    opacity="0.9"
                    onError={(e) => {
                      // Cacher l'image si elle ne charge pas
                      (e.target as SVGImageElement).style.display = 'none';
                    }}
                  />
                )}
              </g>
            );
          })
        )}

        {/* Villes avec images */}
        {cities.map((city, idx) => {
          const [x, y] = city.pos;
          const playerColor = getPlayerColor(city.owner);
          const cityIcon = getCityIcon(city.level, city.owner);
          const iconSize = cellSize * 0.9;
          const iconX = x * cellSize + (cellSize - iconSize) / 2;
          const iconY = y * cellSize + (cellSize - iconSize) / 2;
          
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
                    cx={x * cellSize + cellSize - cellSize * 0.15}
                    cy={y * cellSize + cellSize - cellSize * 0.15}
                    r={cellSize * 0.12}
                    fill={playerColor}
                    stroke="#fff"
                    strokeWidth="1.5"
                  />
                  <text
                    x={x * cellSize + cellSize - cellSize * 0.15}
                    y={y * cellSize + cellSize - cellSize * 0.15}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fill="white"
                    fontSize={cellSize * 0.15}
                    fontWeight="bold"
                  >
                    {city.level}
                  </text>
                </>
              ) : (
                <>
                  {/* Fallback: cercle coloré si pas d'image */}
                  <circle
                    cx={x * cellSize + cellSize / 2}
                    cy={y * cellSize + cellSize / 2}
                    r={cellSize * 0.3}
                    fill={playerColor}
                    stroke="#fff"
                    strokeWidth="2"
                  />
                  <text
                    x={x * cellSize + cellSize / 2}
                    y={y * cellSize + cellSize / 2}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fill="white"
                    fontSize={cellSize * 0.3}
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
          const playerColor = getPlayerColor(unit.owner);
          const unitIcon = getUnitIcon(unit.type, unit.owner);
          const iconSize = cellSize * 0.8;
          const iconX = x * cellSize + (cellSize - iconSize) / 2;
          const iconY = y * cellSize + (cellSize - iconSize) / 2;
          
          return (
            <g key={`unit-${idx}`} className="unit-animation">
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
                    cx={x * cellSize + cellSize - cellSize * 0.2}
                    cy={y * cellSize + cellSize - cellSize * 0.2}
                    r={cellSize * 0.15}
                    fill={playerColor}
                    stroke="#fff"
                    strokeWidth="1.5"
                  />
                  <text
                    x={x * cellSize + cellSize - cellSize * 0.2}
                    y={y * cellSize + cellSize - cellSize * 0.2}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fill="white"
                    fontSize={cellSize * 0.18}
                    fontWeight="bold"
                  >
                    {unit.hp}
                  </text>
                </>
              ) : (
                <>
                  {/* Fallback: cercle coloré si pas d'image */}
                  <circle
                    cx={x * cellSize + cellSize / 2}
                    cy={y * cellSize + cellSize / 2}
                    r={cellSize * 0.25}
                    fill={playerColor}
                    stroke="#fff"
                    strokeWidth="2"
                  />
                  <text
                    x={x * cellSize + cellSize / 2}
                    y={y * cellSize + cellSize / 2 + cellSize * 0.15}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fill="white"
                    fontSize={cellSize * 0.2}
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

