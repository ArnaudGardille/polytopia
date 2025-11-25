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
  // Tier 1
  {
    id: TechType.CLIMBING,
    name: 'Escalade',
    cost: 4, // Coût de base (sera dynamique selon villes)
    description: "Permet aux unités terrestres d'entrer sur les montagnes.",
    dependencies: [],
    unlocks: ['Traversée des montagnes', 'Bonus défense montagne', 'Prérequis pour Navigation et Méditation'],
  },
  {
    id: TechType.FISHING,
    name: 'Pêche',
    cost: 4,
    description: "Débloque la pêche et permet la construction de ports.",
    dependencies: [],
    unlocks: ['Action Pêche', 'Construction de ports', 'Transformation en radeau', 'Prérequis pour Ramming'],
  },
  {
    id: TechType.HUNTING,
    name: 'Chasse',
    cost: 4,
    description: "Débloque la chasse des animaux sauvages.",
    dependencies: [],
    unlocks: ['Action Chasse', 'Prérequis pour Archerie et Sylviculture'],
  },
  {
    id: TechType.ORGANIZATION,
    name: 'Organisation',
    cost: 4,
    description: "Débloque la récolte de fruits et révèle les cultures.",
    dependencies: [],
    unlocks: ['Récolte de fruits', 'Vision des cultures', 'Prérequis pour Agriculture et Stratégie'],
  },
  {
    id: TechType.RIDING,
    name: 'Équitation',
    cost: 4,
    description: "Débloque l'unité Rider.",
    dependencies: [],
    unlocks: ['Unité Rider', 'Prérequis pour Routes et Esprit Libre'],
  },
  // Tier 2
  {
    id: TechType.ARCHERY,
    name: 'Archerie',
    cost: 4, // Coût de base (sera dynamique selon villes)
    description: "Débloque l'unité Archer et donne un bonus de défense dans les forêts.",
    dependencies: [TechType.HUNTING],
    unlocks: ['Unité Archer', 'Bonus défense forêt'],
  },
  {
    id: TechType.RAMMING,
    name: 'Éperonnage',
    cost: 4,
    description: "Débloque l'unité Rammer.",
    dependencies: [TechType.FISHING],
    unlocks: ['Unité Rammer', 'Prérequis pour Aquatisme'],
  },
  {
    id: TechType.FARMING,
    name: 'Agriculture',
    cost: 4,
    description: "Débloque la construction de fermes.",
    dependencies: [TechType.ORGANIZATION],
    unlocks: ['Construction de fermes', 'Gain de population'],
  },
  {
    id: TechType.FORESTRY,
    name: 'Sylviculture',
    cost: 4,
    description: "Débloque la construction de huttes et l'action Couper Forêt.",
    dependencies: [TechType.HUNTING],
    unlocks: ['Construction de huttes', 'Action Couper Forêt'],
  },
  {
    id: TechType.FREE_SPIRIT,
    name: 'Esprit Libre',
    cost: 4,
    description: "Débloque la construction de temples et l'action Disband.",
    dependencies: [TechType.RIDING],
    unlocks: ['Construction de temples', 'Action Disband'],
  },
  {
    id: TechType.MEDITATION,
    name: 'Méditation',
    cost: 4,
    description: "Débloque la construction du Temple de Montagne.",
    dependencies: [TechType.CLIMBING],
    unlocks: ['Temple de Montagne', 'Prérequis pour Philosophie'],
  },
  {
    id: TechType.MINING,
    name: 'Exploitation minière',
    cost: 4,
    description: 'Débloque la construction de mines sur les montagnes.',
    dependencies: [],
    unlocks: ['Construction de mines', 'Gain de population via mines'],
  },
  {
    id: TechType.ROADS,
    name: 'Routes',
    cost: 4,
    description: "Débloque la construction de routes.",
    dependencies: [TechType.RIDING],
    unlocks: ['Construction de routes', 'Réduction coût mouvement'],
  },
  {
    id: TechType.SAILING,
    name: 'Navigation',
    cost: 4,
    description: "Autorise l'embarquement sur les eaux peu profondes et l'océan.",
    dependencies: [TechType.CLIMBING],
    unlocks: ['Transformation en radeau', 'Déplacements en eau peu profonde et océan'],
  },
  {
    id: TechType.STRATEGY,
    name: 'Stratégie',
    cost: 4,
    description: "Débloque l'unité Defender.",
    dependencies: [TechType.ORGANIZATION],
    unlocks: ['Unité Defender'],
  },
  // Tier 3
  {
    id: TechType.AQUATISM,
    name: 'Aquatisme',
    cost: 4, // Coût de base (sera dynamique selon villes)
    description: 'Permet de construire le Temple de l\'eau et donne un bonus de défense sur l\'eau.',
    dependencies: [TechType.RAMMING],
    unlocks: ['Temple de l\'eau', 'Bonus défense eau'],
  },
  {
    id: TechType.PHILOSOPHY,
    name: 'Philosophie',
    cost: 4,
    description: "Réduit le coût des recherches suivantes de 33%.",
    dependencies: [TechType.MEDITATION],
    unlocks: ['Réduction coût technologies (-33%)', 'Unité Mind Bender'],
  },
];
