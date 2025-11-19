# Frontend Polytopia

Interface web moderne pour visualiser les replays de parties Polytopia-JAX et jouer des parties en direct.

## Technologies

- **React 18** avec TypeScript
- **Vite** pour le build et le développement
- **Tailwind CSS** pour le styling
- **SVG** pour le rendu du plateau (optimisé mobile)

## Installation

```bash
cd frontend
npm install
```

## Développement

```bash
npm run dev
```

L'application sera accessible sur `http://localhost:5173` (ou un autre port si 5173 est occupé).

**Important** : Assurez-vous que le backend FastAPI est lancé sur `http://localhost:8000` :

```bash
# Depuis la racine du projet
python scripts/run_web_demo.py
# ou
uvicorn polytopia_jax.web.api:app --reload
```

## Build de production

```bash
npm run build
```

Les fichiers seront générés dans le dossier `dist/`.

## Prévisualisation du build

```bash
npm run preview
```

## Structure

```
frontend/
├── src/
│   ├── components/           # Composants React
│   │   ├── Board.tsx         # Affichage du plateau SVG
│   │   ├── HUD.tsx           # Interface utilisateur pour replays (contrôles, infos)
│   │   ├── GameList.tsx      # Liste des replays disponibles
│   │   ├── MainMenu.tsx      # Menu principal
│   │   ├── ModeSelectionMenu.tsx  # Sélection du mode de jeu
│   │   ├── GameSetupMenu.tsx # Configuration de la partie
│   │   └── LiveGameView.tsx  # Vue pour jouer une partie en direct
│   ├── api.ts                # Client API pour FastAPI
│   ├── types.ts              # Types TypeScript
│   ├── utils/                # Utilitaires
│   │   ├── iconMapper.ts     # Mapping des icônes terrain/unités
│   │   └── actionEncoder.ts  # Encodage des actions de jeu
│   └── styles/               # Styles CSS globaux
├── public/
│   └── icons/                # Icônes Polytopia (terrain, unités)
└── package.json
```

## Menus et Navigation

L'application dispose d'un système de navigation avec plusieurs écrans :

### Menu Principal (`MainMenu`)
Point d'entrée de l'application avec 4 options principales :
- **NEW GAME** → Redirige vers la sélection de mode
- **RESUME GAME** → ⚠️ Non implémenté (à venir)
- **MULTIPLAYER** → ⚠️ Non implémenté (à venir)
- **REPLAY** → Visualisation des replays sauvegardés

Le menu inclut également des boutons de navigation en bas :
- **Settings** → ⚠️ Non implémenté
- **High Score** → ⚠️ Non implémenté
- **Throne Room** → ⚠️ Non implémenté
- **About** → ⚠️ Non implémenté

### Sélection de Mode (`ModeSelectionMenu`)
Permet de choisir le mode de jeu :
- **PERFECTION** → Mode classique avec limite de 30 tours
- **DOMINATION** → ⚠️ Non implémenté (à venir)
- **CREATIVE** → ⚠️ Non implémenté (à venir)

### Configuration de Partie (`GameSetupMenu`)
Menu de configuration avant de démarrer une partie :
- Sélection du nombre d'opposants (3-9)
- Sélection de la difficulté (easy, normal, hard, crazy)
- Affichage des paramètres calculés (taille de carte, limite de tours)
- ⚠️ Le bouton "START GAME" n'est pas encore connecté au backend

### Visualisation de Replays (`game` screen)
Écran complet pour visualiser les replays :
- Liste des replays disponibles
- Navigation entre les tours (précédent/suivant)
- Mode auto-play
- Affichage des informations de jeu (tour, joueur actif, etc.)

### Vue de Partie Live (`LiveGameView`)
Composant pour jouer une partie en direct :
- Affichage du plateau interactif
- Sélection et déplacement d'unités
- Système d'attaque
- Fin de tour
- ⚠️ Non encore intégré dans le flux de navigation principal

## Fonctionnalités Implémentées

### ✅ Complètement Fonctionnel
- 📋 **Menu principal** avec navigation vers les différents écrans
- 🎮 **Sélection de mode** (UI complète, seul PERFECTION est supporté côté backend)
- ⚙️ **Configuration de partie** (UI complète, pas encore connectée)
- 📺 **Visualisation de replays** :
  - Liste des replays disponibles
  - Visualisation du plateau de jeu (SVG)
  - Navigation entre les tours (précédent/suivant)
  - Mode auto-play pour visualiser la partie
  - Affichage des informations de jeu (tour, joueur actif, etc.)
- 🎨 **Interface moderne et responsive** avec design inspiré de Polytopia
- 🔌 **API client** pour les parties live (création, récupération d'état, actions, fin de tour)
- 🎯 **Composant LiveGameView** avec toutes les fonctionnalités de jeu

### 🚧 Partiellement Implémenté
- **GameSetupMenu** : L'interface est complète mais le démarrage de partie n'est pas encore connecté au backend
- **LiveGameView** : Le composant est fonctionnel mais n'est pas encore intégré dans le flux de navigation de `App.tsx`

### ❌ À Implémenter
- **RESUME GAME** : Reprendre une partie sauvegardée
- **MULTIPLAYER** : Mode multijoueur
- **DOMINATION** : Mode de jeu domination (backend + frontend)
- **CREATIVE** : Mode créatif (backend + frontend)
- **Settings** : Menu de paramètres
- **High Score** : Affichage des meilleurs scores
- **Throne Room** : Salle du trône (statistiques)
- **About** : Page à propos
- **Intégration LiveGameView** : Connecter le flux de navigation pour démarrer et jouer une partie live
- **Sauvegarde de parties** : Système pour sauvegarder et reprendre des parties en cours

## Configuration

### Proxy API

Le proxy vers l'API FastAPI est configuré dans `vite.config.ts`. En développement, les requêtes vers `/games` sont automatiquement redirigées vers `http://localhost:8000`.

### Variables d'environnement

Créez un fichier `.env` pour configurer l'URL de l'API :

```env
VITE_API_URL=http://localhost:8000
```

## Icônes Polytopia

Les icônes sont stockées dans `public/icons/`. Voir `public/icons/README.md` pour plus d'informations sur la récupération des vraies icônes Polytopia.

## Conversion en app mobile

Ce frontend est optimisé pour une future conversion en app iPhone via **Capacitor** :

1. Installer Capacitor : `npm install @capacitor/core @capacitor/cli`
2. Initialiser : `npx cap init`
3. Ajouter la plateforme iOS : `npx cap add ios`
4. Build et sync : `npm run build && npx cap sync`

## Architecture de Navigation

Le système de navigation utilise un état `currentScreen` dans `App.tsx` qui peut prendre les valeurs suivantes :
- `'mainMenu'` → Affiche le menu principal
- `'modeSelection'` → Affiche la sélection de mode
- `'gameSetup'` → Affiche la configuration de partie
- `'game'` → Affiche la visualisation de replays

**Note** : Le composant `LiveGameView` existe mais n'est pas encore intégré dans ce système de navigation. Il faudra ajouter un nouvel écran `'liveGame'` pour l'intégrer complètement.

## Prochaines Étapes

1. **Connecter GameSetupMenu au backend** : Implémenter la logique de démarrage de partie dans `App.tsx` pour appeler `createPerfectionGame()` et naviguer vers `LiveGameView`
2. **Intégrer LiveGameView** : Ajouter un nouvel écran `'liveGame'` dans le système de navigation
3. **Implémenter RESUME GAME** : Créer un système de sauvegarde/chargement de parties
4. **Ajouter les autres modes** : Implémenter DOMINATION et CREATIVE côté backend et frontend
5. **Implémenter les menus secondaires** : Settings, High Score, Throne Room, About

## Support

Pour toute question ou problème, consultez la documentation principale du projet dans le README à la racine.

