// Définitions complètes des unités (correspondant à rules.py dans le backend)

import { UnitType, TechType } from '../types';

export interface UnitDefinition {
  id: number;
  name: string;
  hp: number;
  attack: number;
  defense: number;
  movement: number;
  range: number;
  cost: number;
  requiredTech: number;
  isNaval: boolean;
  canPromote: boolean;
  description: string;
}

// Statistiques des unités selon rules.py
export const UNIT_DEFINITIONS: Record<number, UnitDefinition> = {
  [UnitType.WARRIOR]: {
    id: UnitType.WARRIOR,
    name: 'Guerrier',
    hp: 10,
    attack: 2,
    defense: 2,
    movement: 1,
    range: 1,
    cost: 2,
    requiredTech: TechType.NONE,
    isNaval: false,
    canPromote: true,
    description: 'Unité de base polyvalente',
  },
  [UnitType.DEFENDER]: {
    id: UnitType.DEFENDER,
    name: 'Défenseur',
    hp: 15,
    attack: 1,
    defense: 3,
    movement: 1,
    range: 1,
    cost: 3,
    requiredTech: TechType.STRATEGY,
    isNaval: false,
    canPromote: true,
    description: 'Unité défensive robuste',
  },
  [UnitType.ARCHER]: {
    id: UnitType.ARCHER,
    name: 'Archer',
    hp: 8,
    attack: 2,
    defense: 1,
    movement: 1,
    range: 2,
    cost: 3,
    requiredTech: TechType.ARCHERY,
    isNaval: false,
    canPromote: true,
    description: 'Attaque à distance',
  },
  [UnitType.RIDER]: {
    id: UnitType.RIDER,
    name: 'Cavalier',
    hp: 10,
    attack: 3,
    defense: 1,
    movement: 2,
    range: 1,
    cost: 4,
    requiredTech: TechType.RIDING,
    isNaval: false,
    canPromote: true,
    description: 'Unité mobile et offensive',
  },
  [UnitType.RAFT]: {
    id: UnitType.RAFT,
    name: 'Radeau',
    hp: 8,
    attack: 2,
    defense: 1,
    movement: 2,
    range: 1,
    cost: 0, // Créé via embarquement
    requiredTech: TechType.SAILING,
    isNaval: true,
    canPromote: false,
    description: 'Unité navale temporaire',
  },
  [UnitType.KNIGHT]: {
    id: UnitType.KNIGHT,
    name: 'Chevalier',
    hp: 15,
    attack: 4,
    defense: 1,
    movement: 3,
    range: 1,
    cost: 6,
    requiredTech: TechType.CHIVALRY,
    isNaval: false,
    canPromote: true,
    description: 'Unité de cavalerie lourde',
  },
  [UnitType.SWORDSMAN]: {
    id: UnitType.SWORDSMAN,
    name: 'Épéiste',
    hp: 15,
    attack: 3,
    defense: 2,
    movement: 1,
    range: 1,
    cost: 5,
    requiredTech: TechType.SMITHERY,
    isNaval: false,
    canPromote: true,
    description: 'Guerrier amélioré',
  },
  [UnitType.CATAPULT]: {
    id: UnitType.CATAPULT,
    name: 'Catapulte',
    hp: 8,
    attack: 4,
    defense: 1,
    movement: 1,
    range: 3,
    cost: 6,
    requiredTech: TechType.MATHEMATICS,
    isNaval: false,
    canPromote: true,
    description: 'Siège à longue portée',
  },
  [UnitType.GIANT]: {
    id: UnitType.GIANT,
    name: 'Géant',
    hp: 40,
    attack: 5,
    defense: 3,
    movement: 1,
    range: 1,
    cost: 20,
    requiredTech: TechType.NONE, // Obtenu via amélioration de ville
    isNaval: false,
    canPromote: false,
    description: 'Super unité puissante',
  },
};

// Liste des unités entraînables (exclut RAFT et GIANT qui sont obtenus autrement)
export const TRAINABLE_UNITS = [
  UnitType.WARRIOR,
  UnitType.DEFENDER,
  UnitType.ARCHER,
  UnitType.RIDER,
  UnitType.KNIGHT,
  UnitType.SWORDSMAN,
  UnitType.CATAPULT,
];

/**
 * Retourne le nom d'une unité à partir de son type
 */
export function getUnitName(unitType: number): string {
  return UNIT_DEFINITIONS[unitType]?.name ?? `Unité #${unitType}`;
}

/**
 * Retourne la définition complète d'une unité
 */
export function getUnitDefinition(unitType: number): UnitDefinition | undefined {
  return UNIT_DEFINITIONS[unitType];
}

/**
 * Vérifie si un type d'unité peut être entraîné (hors RAFT et GIANT)
 */
export function isTrainable(unitType: number): boolean {
  return (TRAINABLE_UNITS as readonly number[]).includes(unitType);
}

