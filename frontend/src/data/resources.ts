import { ResourceType, TechType } from '../types';

export interface ResourceDefinition {
  id: number;
  name: string;
  shortLabel: string;
  description: string;
  cost: number;
  population: number;
  requiredTech?: number;
  color: string;
}

export const RESOURCE_DEFINITIONS: Record<number, ResourceDefinition> = {
  [ResourceType.NONE]: {
    id: ResourceType.NONE,
    name: 'Aucune',
    shortLabel: '',
    description: '',
    cost: 0,
    population: 0,
    color: '#9ca3af',
  },
  [ResourceType.FRUIT]: {
    id: ResourceType.FRUIT,
    name: 'Fruit',
    shortLabel: 'Fr',
    description: '+1 population dans la ville adjacente.',
    cost: 2,
    population: 1,
    color: '#f97316',
  },
  [ResourceType.FISH]: {
    id: ResourceType.FISH,
    name: 'Poisson',
    shortLabel: 'Po',
    description: '+1 population en récoltant en mer.',
    cost: 2,
    population: 1,
    requiredTech: TechType.SAILING,
    color: '#3b82f6',
  },
  [ResourceType.ORE]: {
    id: ResourceType.ORE,
    name: 'Minerai',
    shortLabel: 'Mn',
    description: '+2 population en construisant une mine.',
    cost: 4,
    population: 2,
    requiredTech: TechType.MINING,
    color: '#d1d5db',
  },
};

export const HARVEST_ZONE_OFFSETS: [number, number][] = [
  [-1, -1],
  [0, -1],
  [1, -1],
  [-1, 0],
  [1, 0],
  [-1, 1],
  [0, 1],
  [1, 1],
];

export function getResourceDefinition(
  type?: number | null
): ResourceDefinition | undefined {
  if (typeof type !== 'number') {
    return undefined;
  }
  return RESOURCE_DEFINITIONS[type];
}
