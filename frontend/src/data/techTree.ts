import { TechType } from '../types';

export interface TechnologyNode {
  id: number;
  name: string;
  cost: number;
  description: string;
  dependencies: number[];
  unlocks: string[];
}

export const TECHNOLOGY_TREE: TechnologyNode[] = [
  {
    id: TechType.CLIMBING,
    name: 'Escalade',
    cost: 3,
    description: "Permet aux unites terrestres d'entrer sur les montagnes.",
    dependencies: [],
    unlocks: ['Traversee des montagnes', 'Prerequis pour Navigation'],
  },
  {
    id: TechType.SAILING,
    name: 'Navigation',
    cost: 4,
    description:
      "Autorise la construction de ports et l'embarquement sur les eaux peu profondes.",
    dependencies: [TechType.CLIMBING],
    unlocks: ['Construction de ports', 'Transformation en radeau', 'Deplacements en eau peu profonde'],
  },
  {
    id: TechType.MINING,
    name: 'Exploitation miniere',
    cost: 3,
    description: 'Debloque la construction de mines sur les montagnes.',
    dependencies: [],
    unlocks: ['Construction de mines', 'Gain de population via mines'],
  },
];
