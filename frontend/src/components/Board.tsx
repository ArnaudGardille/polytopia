import { useMemo, useEffect, useState, useRef, useCallback, type MouseEvent } from 'react';
import type { GameStateView, CityView } from '../types';
import { ResourceType } from '../types';
import { getTerrainIcon, getPlayerColor, getUnitIcon, getCityIcon } from '../utils/iconMapper';
import { HARVEST_ZONE_OFFSETS } from '../data/resources';

// Proportions tirées du sprite haute résolution (Grass.png)
const BASE_TILE_WIDTH = 2067;
const BASE_TILE_HEIGHT = 2249;
const TILE_HEIGHT_OVER_WIDTH = BASE_TILE_HEIGHT / BASE_TILE_WIDTH; // ≈ 1.088

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

// Fonction pour calculer la position isométrique d'une case
// Transformation isométrique simple : les diagonales visuelles correspondent aux diagonales logiques
function hexToPixel(x: number, y: number, tileWidth: number): [number, number] {
  const tileHeight = tileWidth * TILE_HEIGHT_OVER_WIDTH;
  // Transformation isométrique : rotation de 45° avec compression verticale
  // Les diagonales visuelles correspondent aux diagonales logiques
  const pixelX = (x - y) * tileWidth / 2;
  const pixelY = (x + y) * tileHeight / 4; // Distance verticale divisée par 2
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
  
  // États pour zoom et pan
  const [zoom, setZoom] = useState(1.0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [hasPanned, setHasPanned] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);
  
  const MIN_ZOOM = 0.5;
  const MAX_ZOOM = 3.0;

  // Calculer la taille de cellule responsive pour rendu isométrique
  const hexSize = useMemo(() => {
    if (propCellSize) return propCellSize;
    const maxWidth = Math.min(containerSize.width - 40, 1600);
    const maxHeight = containerSize.height - 40;
    const tileHeightUnit = TILE_HEIGHT_OVER_WIDTH;
    // Pour un rendu isométrique, la largeur totale dépend de (width + height) / 2
    // et la hauteur totale dépend de (width + height) / 2 aussi
    const diagonalSize = Math.max(width, height);
    const widthDenominator = diagonalSize > 0 ? diagonalSize : 1;
    const heightDenominator = diagonalSize > 0 ? diagonalSize * tileHeightUnit : tileHeightUnit;
    const cellSizeByWidth = maxWidth / widthDenominator;
    const cellSizeByHeight = maxHeight / heightDenominator;
    return Math.min(cellSizeByWidth, cellSizeByHeight, 160);
  }, [width, height, containerSize, propCellSize]);

  const tileWidth = hexSize;
  const tileHeight = tileWidth * TILE_HEIGHT_OVER_WIDTH;
  // Pas de correction nécessaire pour rendu isométrique pur
  const spriteCenterCorrectionY = 0;

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

  // Calculer les dimensions du viewBox pour rendu isométrique
  const viewBox = useMemo(() => {
    const paddingX = tileWidth * 0.05;
    const paddingY = tileHeight * 0.05;
    // Pour un rendu isométrique, les dimensions dépendent de la transformation (x-y) et (x+y)
    // Largeur max : de (0, height-1) à (width-1, 0) = (width-1) - (height-1) en x transformé
    // Hauteur max : de (0, 0) à (width-1, height-1) = (width-1) + (height-1) en y transformé (divisé par 4)
    const maxX = width > 0 ? (width - 1) * tileWidth / 2 : 0;
    const minX = height > 0 ? -(height - 1) * tileWidth / 2 : 0;
    const boardWidth = maxX - minX + tileWidth;
    const boardHeight = height > 0 && width > 0 ? ((width - 1) + (height - 1)) * tileHeight / 4 + tileHeight : tileHeight;
    const viewMinX = minX - paddingX;
    const viewMinY = -paddingY;
    const viewWidth = boardWidth + paddingX * 2;
    const viewHeight = boardHeight + paddingY * 2;
    return `${viewMinX} ${viewMinY} ${viewWidth} ${viewHeight}`;
  }, [width, height, tileWidth, tileHeight]);

  // Calculer les valeurs du viewBox pour le rectangle de fond
  const viewBoxData = useMemo(() => {
    const paddingX = tileWidth * 0.05;
    const paddingY = tileHeight * 0.05;
    const maxX = width > 0 ? (width - 1) * tileWidth / 2 : 0;
    const minX = height > 0 ? -(height - 1) * tileWidth / 2 : 0;
    const boardWidth = maxX - minX + tileWidth;
    const boardHeight = height > 0 && width > 0 ? ((width - 1) + (height - 1)) * tileHeight / 4 + tileHeight : tileHeight;
    return {
      x: minX - paddingX,
      y: -paddingY,
      width: boardWidth + paddingX * 2,
      height: boardHeight + paddingY * 2,
    };
  }, [width, height, tileWidth, tileHeight]);

  const moveTargetsMap = useMemo(() => {
    if (!moveTargets || moveTargets.length === 0) return null;
    const map = new Map<string, MoveTarget>();
    moveTargets.forEach((target) => {
      map.set(`${target.x}-${target.y}`, target);
    });
    return map;
  }, [moveTargets]);

  const handleBackgroundClick = useCallback(() => {
    // Ne pas désélectionner si on vient de faire un pan
    if (hasPanned) {
      setHasPanned(false);
      return;
    }
    
    if (onDeselectUnit) {
      onDeselectUnit();
    }
    if (onDeselectCity) {
      onDeselectCity();
    }
  }, [hasPanned, onDeselectUnit, onDeselectCity]);

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

  // Handler pour commencer le pan sur le rectangle de fond ou les éléments de terrain
  const handleBackgroundMouseDown = useCallback((event: MouseEvent<SVGElement>) => {
    // Ne pas démarrer le pan si on clique sur un élément interactif
    const target = event.target as SVGElement;
    // Vérifier si on clique sur un élément interactif (unités, villes, moveTargets)
    const isInteractive = 
      target.closest('.unit-animation') !== null ||
      target.closest('.city-marker') !== null ||
      target.closest('g[key*="movetarget"]') !== null ||
      (target.tagName === 'circle' && target.closest('g[key*="movetarget"]') !== null) ||
      (target.tagName === 'g' && target.getAttribute('key')?.includes('movetarget')) ||
      target.parentElement?.getAttribute('key')?.includes('movetarget') === true;
    
    if (!isInteractive) {
      event.stopPropagation();
      setIsPanning(true);
      setHasPanned(false);
      setPanStart({ x: event.clientX, y: event.clientY });
    } else {
      // Si c'est un élément interactif, ne pas démarrer le pan
      event.stopPropagation();
    }
  }, []);

  // Handler pour le mouvement pendant le pan
  const handleMouseMove = useCallback((event: MouseEvent<SVGSVGElement>) => {
    if (!isPanning) return;
    
    const deltaX = event.clientX - panStart.x;
    const deltaY = event.clientY - panStart.y;
    
    // Si on a bougé significativement, marquer qu'on a fait un pan
    if (Math.abs(deltaX) > 2 || Math.abs(deltaY) > 2) {
      setHasPanned(true);
    }
    
    // Convertir le delta en coordonnées SVG
    if (!svgRef.current) return;
    const svg = svgRef.current;
    const rect = svg.getBoundingClientRect();
    const viewBox = svg.viewBox.baseVal;
    const svgDeltaX = (deltaX / rect.width) * viewBox.width;
    const svgDeltaY = (deltaY / rect.height) * viewBox.height;
    
    setPan(prev => ({
      x: prev.x + svgDeltaX,
      y: prev.y + svgDeltaY,
    }));
    setPanStart({ x: event.clientX, y: event.clientY });
  }, [isPanning, panStart]);

  // Handler pour terminer le pan
  const handleMouseUp = useCallback(() => {
    setIsPanning(false);
  }, []);

  // Handler pour quand la souris sort du SVG
  const handleMouseLeave = useCallback(() => {
    setIsPanning(false);
  }, []);

  // Ajouter les event listeners wheel avec passive: false pour pouvoir utiliser preventDefault
  useEffect(() => {
    const svg = svgRef.current;
    const container = svg?.parentElement;
    
    if (!svg || !container) return;

    const handleSvgWheel = (event: WheelEvent) => {
      event.preventDefault();
      event.stopPropagation();
      if (!svg) return;

      // Ralentir le zoom : facteur plus petit et basé sur deltaY
      const zoomSpeed = 0.001;
      const delta = 1 - event.deltaY * zoomSpeed;
      
      setZoom((currentZoom) => {
        const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, currentZoom * delta));
        
        if (newZoom === currentZoom) return currentZoom;

        // Convertir la position de la souris en coordonnées SVG avant transformation
        const rect = svg.getBoundingClientRect();
        const viewBox = svg.viewBox.baseVal;
        const mouseX = ((event.clientX - rect.left) / rect.width) * viewBox.width + viewBox.x;
        const mouseY = ((event.clientY - rect.top) / rect.height) * viewBox.height + viewBox.y;
        
        // Calculer le nouveau pan pour zoomer vers la position de la souris
        setPan((currentPan) => {
          const zoomFactor = newZoom / currentZoom;
          const newPanX = mouseX - (mouseX - currentPan.x) * zoomFactor;
          const newPanY = mouseY - (mouseY - currentPan.y) * zoomFactor;
          return { x: newPanX, y: newPanY };
        });
        
        return newZoom;
      });
    };

    const handleContainerWheel = (event: WheelEvent) => {
      // Si l'événement vient du SVG, ne rien faire (il sera géré par handleSvgWheel)
      if (event.target === svg || svg.contains(event.target as Node)) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
    };

    // Ajouter les listeners avec passive: false
    svg.addEventListener('wheel', handleSvgWheel, { passive: false });
    container.addEventListener('wheel', handleContainerWheel, { passive: false });

    return () => {
      svg.removeEventListener('wheel', handleSvgWheel);
      container.removeEventListener('wheel', handleContainerWheel);
    };
  }, []);

  return (
    <div 
      className="board-container" 
      onClick={handleBackgroundClick}
      style={{ 
        cursor: isPanning ? 'grabbing' : 'grab',
        userSelect: isPanning ? 'none' : 'auto',
      }}
    >
      <svg
        ref={svgRef}
        className="board-svg"
        viewBox={viewBox}
        xmlns="http://www.w3.org/2000/svg"
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        style={{
          cursor: isPanning ? 'grabbing' : 'grab',
        }}
      >
        <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
          {/* Rectangle de fond invisible pour gérer le pan */}
          <rect
            x={viewBoxData.x}
            y={viewBoxData.y}
            width={viewBoxData.width}
            height={viewBoxData.height}
            fill="transparent"
            onMouseDown={handleBackgroundMouseDown}
            style={{
              cursor: isPanning ? 'grabbing' : 'grab',
            }}
          />
        {/* Grille de terrain */}
        {terrain.map((row, y) =>
          row.map((terrainType, x) => {
            const terrainIcon = getTerrainIcon(terrainType);
            const [centerX, centerY] = hexToPixel(x, y, tileWidth);
            const imageX = centerX - tileWidth / 2;
            const imageY = centerY - tileHeight / 2;
            const inHarvestZone = harvestZone?.has(`${x}-${y}`) ?? false;
            
            // Vérifier la visibilité de la case
            const visibilityMask = state.visibility_mask;
            const isVisible = visibilityMask 
              ? (visibilityMask[y]?.[x] === 1)
              : true; // Par défaut, tout visible si pas de masque
            
            return (
              <g 
                key={`terrain-${x}-${y}`}
              >
                {/* Rectangle invisible pour permettre le pan sur cette case */}
                {/* Ne pas ajouter de rectangle si cette case a un moveTarget (les moveTargets sont rendus après) */}
                {!moveTargetsMap?.has(`${x}-${y}`) && (
                  <rect
                    x={imageX}
                    y={imageY}
                    width={tileWidth}
                    height={tileHeight}
                    fill="transparent"
                    onMouseDown={(event) => {
                      // Ne pas démarrer le pan si on clique sur un élément interactif
                      const target = event.target as SVGElement;
                      const isInteractive = 
                        target.closest('.unit-animation') !== null ||
                        target.closest('.city-marker') !== null;
                      
                      if (!isInteractive) {
                        setIsPanning(true);
                        setHasPanned(false);
                        setPanStart({ x: event.clientX, y: event.clientY });
                      }
                    }}
                    style={{ cursor: isPanning ? 'grabbing' : 'grab' }}
                  />
                )}
                {terrainIcon && (
                  <image
                    href={terrainIcon}
                    x={imageX}
                    y={imageY}
                    width={tileWidth}
                    height={tileHeight}
                    preserveAspectRatio="xMidYMid meet"
                    opacity={isVisible ? "0.9" : "0.3"}
                    pointerEvents="none"
                    onError={(e) => {
                      (e.target as SVGImageElement).style.display = 'none';
                    }}
                  />
                )}
                {!isVisible && (
                  <image
                    href="/icons/terrain/Clouds.png"
                    x={imageX}
                    y={imageY}
                    width={tileWidth}
                    height={tileHeight}
                    preserveAspectRatio="xMidYMid meet"
                    opacity="0.9"
                    pointerEvents="none"
                    onError={(e) => {
                      (e.target as SVGImageElement).style.display = 'none';
                    }}
                  />
                )}
                {inHarvestZone && isVisible && (
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
                {/* Afficher les routes */}
                {isVisible && state.has_road?.[y]?.[x] && (
                  <rect
                    x={centerX - tileWidth * 0.15}
                    y={centerY - tileHeight * 0.02}
                    width={tileWidth * 0.3}
                    height={tileHeight * 0.04}
                    fill="#8B7355"
                    stroke="#5C4A33"
                    strokeWidth={1}
                    opacity={0.9}
                    pointerEvents="none"
                  />
                )}
                {/* Afficher les ponts */}
                {isVisible && state.has_bridge?.[y]?.[x] && (
                  <g pointerEvents="none">
                    <rect
                      x={centerX - tileWidth * 0.2}
                      y={centerY - tileHeight * 0.03}
                      width={tileWidth * 0.4}
                      height={tileHeight * 0.06}
                      fill="#A0522D"
                      stroke="#654321"
                      strokeWidth={1}
                      opacity={0.95}
                    />
                    <line
                      x1={centerX - tileWidth * 0.18}
                      y1={centerY - tileHeight * 0.06}
                      x2={centerX - tileWidth * 0.18}
                      y2={centerY + tileHeight * 0.06}
                      stroke="#654321"
                      strokeWidth={2}
                    />
                    <line
                      x1={centerX + tileWidth * 0.18}
                      y1={centerY - tileHeight * 0.06}
                      x2={centerX + tileWidth * 0.18}
                      y2={centerY + tileHeight * 0.06}
                      stroke="#654321"
                      strokeWidth={2}
                    />
                  </g>
                )}
                {/* Afficher les ruines */}
                {isVisible && state.has_ruin?.[y]?.[x] && (
                  <g pointerEvents="none">
                    <circle
                      cx={centerX}
                      cy={centerY}
                      r={tileWidth * 0.15}
                      fill="rgba(139, 69, 19, 0.7)"
                      stroke="#8B4513"
                      strokeWidth={2}
                    />
                    <text
                      x={centerX}
                      y={centerY}
                      textAnchor="middle"
                      dominantBaseline="middle"
                      fill="#FFD700"
                      fontSize={tileWidth * 0.15}
                      fontWeight="bold"
                    >
                      ?
                    </text>
                  </g>
                )}
                {/* Afficher indicateur de ressource (si non affiché par le terrain) */}
                {isVisible && state.resource_type?.[y]?.[x] !== undefined && 
                  state.resource_type[y][x] !== ResourceType.NONE && (
                  <g pointerEvents="none">
                    {/* Indicateur petit en bas à gauche de la case */}
                    <circle
                      cx={centerX - tileWidth * 0.25}
                      cy={centerY + tileHeight * 0.15}
                      r={tileWidth * 0.08}
                      fill={state.resource_available?.[y]?.[x] ? '#22c55e' : '#6b7280'}
                      stroke="#fff"
                      strokeWidth={1}
                      opacity={0.9}
                    />
                  </g>
                )}
                {isVisible && (
                  <text
                    x={centerX}
                    y={centerY - tileHeight * 0.35}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fill="white"
                    fontSize={Math.max(4, Math.min(tileWidth, tileHeight) * 0.04)}
                    fontWeight="bold"
                    stroke="black"
                    strokeWidth={Math.max(0.15, Math.min(tileWidth, tileHeight) * 0.008)}
                    paintOrder="stroke"
                    pointerEvents="none"
                    style={{ userSelect: 'none' }}
                  >
                    {`${x},${y}`}
                  </text>
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
                onMouseDown={(event) => {
                  // Empêcher le pan de démarrer quand on clique sur un moveTarget
                  event.stopPropagation();
                }}
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
                {/* Rectangle invisible pour capturer les événements et afficher le curseur pointer */}
                <rect
                  x={centerX - Math.min(tileWidth, tileHeight) * 0.4}
                  y={centerY - Math.min(tileWidth, tileHeight) * 0.4}
                  width={Math.min(tileWidth, tileHeight) * 0.8}
                  height={Math.min(tileWidth, tileHeight) * 0.8}
                  fill="transparent"
                  pointerEvents="auto"
                  style={{
                    cursor: onSelectMoveTarget ? 'pointer' : 'default',
                  }}
                />
                <circle
                  cx={centerX}
                  cy={centerY}
                  r={Math.min(tileWidth, tileHeight) * 0.35}
                  fill={
                    moveTarget.type === 'attack'
                      ? 'rgba(239,68,68,0.2)'
                      : 'rgba(59,130,246,0.2)'
                  }
                  pointerEvents="none"
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
                  pointerEvents="none"
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
                  pointerEvents="none"
                />
              </g>
            );
          })}

        {/* Villes avec images */}
        {cities
          .filter((city) => {
            // Vérification de sécurité : masquer les villes non visibles
            const visibilityMask = state.visibility_mask;
            if (!visibilityMask) return true; // Pas de masque = tout visible
            const [x, y] = city.pos;
            return visibilityMask[y]?.[x] === 1;
          })
          .map((city, idx) => {
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
                  {/* Indicateurs de bâtiments */}
                  {(() => {
                    const buildingBadges: { color: string; label: string }[] = [];
                    if (state.city_ports?.[y]?.[x]) buildingBadges.push({ color: '#0ea5e9', label: 'P' });
                    if (state.city_has_windmill?.[y]?.[x]) buildingBadges.push({ color: '#84cc16', label: 'W' });
                    if (state.city_has_forge?.[y]?.[x]) buildingBadges.push({ color: '#f97316', label: 'F' });
                    if (state.city_has_sawmill?.[y]?.[x]) buildingBadges.push({ color: '#22c55e', label: 'S' });
                    if (state.city_has_market?.[y]?.[x]) buildingBadges.push({ color: '#eab308', label: 'M' });
                    if (state.city_has_temple?.[y]?.[x]) buildingBadges.push({ color: '#a855f7', label: 'T' });
                    if (state.city_has_monument?.[y]?.[x]) buildingBadges.push({ color: '#ec4899', label: 'O' });
                    if (state.city_has_wall?.[y]?.[x]) buildingBadges.push({ color: '#6b7280', label: '⛊' });
                    if (state.city_has_park?.[y]?.[x]) buildingBadges.push({ color: '#10b981', label: '♣' });
                    
                    const badgeRadius = Math.min(tileWidth, tileHeight) * 0.06;
                    const startX = centerX - (buildingBadges.length - 1) * badgeRadius * 1.5 / 2;
                    const badgeBaseY = iconCenterY - iconSize * 0.35;
                    
                    return buildingBadges.map((badge, i) => (
                      <g key={`building-badge-${i}`}>
                        <circle
                          cx={startX + i * badgeRadius * 1.5}
                          cy={badgeBaseY}
                          r={badgeRadius}
                          fill={badge.color}
                          stroke="#fff"
                          strokeWidth="0.5"
                        />
                        <text
                          x={startX + i * badgeRadius * 1.5}
                          y={badgeBaseY}
                          textAnchor="middle"
                          dominantBaseline="middle"
                          fill="white"
                          fontSize={badgeRadius * 1.2}
                          fontWeight="bold"
                        >
                          {badge.label}
                        </text>
                      </g>
                    ));
                  })()}
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
        {units
          .filter((unit) => {
            // Vérification de sécurité : masquer les unités non visibles
            const visibilityMask = state.visibility_mask;
            if (!visibilityMask) return true; // Pas de masque = tout visible
            const [x, y] = unit.pos;
            return visibilityMask[y]?.[x] === 1;
          })
          .map((unit, idx) => {
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
        </g>
      </svg>
    </div>
  );
}
