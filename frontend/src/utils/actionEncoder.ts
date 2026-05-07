export enum ActionType {
  NO_OP = 0,
  MOVE = 1,
  ATTACK = 2,
  TRAIN_UNIT = 3,
  BUILD = 4,
  RESEARCH_TECH = 5,
  END_TURN = 6,
  HARVEST_RESOURCE = 7,
  RECOVER = 8,
}

export enum Direction {
  UP = 0,
  UP_RIGHT = 1,
  RIGHT = 2,
  DOWN_RIGHT = 3,
  DOWN = 4,
  DOWN_LEFT = 5,
  LEFT = 6,
  UP_LEFT = 7,
}

// Système de grille simple : 8 directions avec deltas {-1, 0, 1} en x et y
const DIRECTION_DELTA: Record<Direction, [number, number]> = {
  [Direction.UP]: [0, -1],
  [Direction.UP_RIGHT]: [1, -1],
  [Direction.RIGHT]: [1, 0],
  [Direction.DOWN_RIGHT]: [1, 1],
  [Direction.DOWN]: [0, 1],
  [Direction.DOWN_LEFT]: [-1, 1],
  [Direction.LEFT]: [-1, 0],
  [Direction.UP_LEFT]: [-1, -1],
};

const DELTA_TO_DIRECTION: Record<string, Direction> = Object.entries(
  DIRECTION_DELTA
).reduce((acc, [dir, delta]) => {
  acc[`${delta[0]},${delta[1]}`] = Number(dir) as Direction;
  return acc;
}, {} as Record<string, Direction>);

// Format d'encodage — doit rester strictement aligné sur
// polytopia_jax/core/actions.py::encode_action :
//   action_type : 4 bits (0..3)
//   unit_id     : 8 bits (4..11)
//   direction   : 3 bits (12..14)
//   target_x    : 5 bits (15..19)
//   target_y    : 5 bits (20..24)
//   unit_type   : 5 bits (25..29)
export function encodeAction(params: {
  actionType: ActionType;
  unitId?: number;
  direction?: Direction;
  targetPos?: [number, number];
  unitType?: number;
}): number {
  const { actionType, unitId, direction, targetPos, unitType } = params;
  let encoded = actionType & 0xf;

  if (typeof unitId === 'number') {
    encoded |= (unitId & 0xff) << 4;
  }

  if (typeof direction === 'number') {
    encoded |= (direction & 0x7) << 12;
  }

  if (targetPos) {
    encoded |= (targetPos[0] & 0x1f) << 15;
    encoded |= (targetPos[1] & 0x1f) << 20;
  }

  if (typeof unitType === 'number') {
    encoded |= (unitType & 0x1f) << 25;
  }

  return encoded;
}

export function encodeMove(unitId: number, direction: Direction): number {
  return encodeAction({ actionType: ActionType.MOVE, unitId, direction });
}

export function encodeAttack(
  unitId: number,
  targetPos: [number, number]
): number {
  return encodeAction({
    actionType: ActionType.ATTACK,
    unitId,
    targetPos,
  });
}

export function encodeTrainUnit(
  unitType: number,
  targetPos: [number, number]
): number {
  return encodeAction({
    actionType: ActionType.TRAIN_UNIT,
    targetPos,
    unitType,
  });
}

export function encodeResearchTech(techId: number): number {
  return encodeAction({
    actionType: ActionType.RESEARCH_TECH,
    unitType: techId,
  });
}

export function encodeHarvestResource(targetPos: [number, number]): number {
  return encodeAction({
    actionType: ActionType.HARVEST_RESOURCE,
    targetPos,
  });
}

export function encodeBuild(
  buildingType: number,
  targetPos: [number, number]
): number {
  return encodeAction({
    actionType: ActionType.BUILD,
    targetPos,
    unitType: buildingType, // buildingType est encodé dans le même champ que unitType
  });
}

export function encodeEndTurn(): number {
  return encodeAction({ actionType: ActionType.END_TURN });
}

export function encodeRecover(unitId: number): number {
  return encodeAction({
    actionType: ActionType.RECOVER,
    unitId,
  });
}

export function getDirectionDelta(direction: Direction): [number, number] {
  return DIRECTION_DELTA[direction];
}

export function directionFromDelta(
  delta: [number, number]
): Direction | null {
  return DELTA_TO_DIRECTION[`${delta[0]},${delta[1]}`] ?? null;
}
