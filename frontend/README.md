# Frontend Polytopia Replay Viewer

Interface web moderne pour visualiser les replays de parties Polytopia-JAX.

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
│   ├── components/      # Composants React
│   │   ├── Board.tsx   # Affichage du plateau SVG
│   │   ├── HUD.tsx     # Interface utilisateur (contrôles, infos)
│   │   └── GameList.tsx # Liste des replays
│   ├── api.ts          # Client API pour FastAPI
│   ├── types.ts        # Types TypeScript
│   ├── utils/          # Utilitaires (mapping icônes, couleurs)
│   └── styles/         # Styles CSS globaux
├── public/
│   └── icons/          # Icônes Polytopia (terrain, unités)
└── package.json
```

## Fonctionnalités

- 📋 Liste des replays disponibles
- 🎮 Visualisation du plateau de jeu
- ⏯️ Navigation entre les tours (précédent/suivant)
- ▶️ Mode auto-play pour visualiser la partie
- 📊 Affichage des informations de jeu (tour, joueur actif, etc.)
- 🎨 Interface moderne et responsive

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

## Support

Pour toute question ou problème, consultez la documentation principale du projet dans le README à la racine.

