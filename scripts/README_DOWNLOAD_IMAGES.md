# Script de téléchargement des images Polytopia

Ce script télécharge automatiquement toutes les images Polytopia depuis le wiki officiel.

## Installation des dépendances

```bash
pip install requests beautifulsoup4
```

Ou si vous utilisez le projet installé :

```bash
pip install -e .
```

## Utilisation

### Téléchargement de base

```bash
python scripts/download_polytopia_images.py
```

Cela téléchargera toutes les images trouvées sur les pages principales du wiki dans `frontend/public/icons/`.

### Options disponibles

```bash
python scripts/download_polytopia_images.py --help
```

Options principales :

- `--output-dir` : Dossier de sortie (défaut: `frontend/public/icons`)
- `--wiki-url` : URL de base du wiki (défaut: `https://polytopia.fandom.com/wiki/The_Battle_of_Polytopia_Wiki`)
- `--max-pages` : Nombre maximum de pages à parcourir (défaut: illimité)
- `--delay` : Délai entre les requêtes en secondes (défaut: 0.5)

### Exemples

Télécharger avec un délai plus long (pour éviter de surcharger le serveur) :

```bash
python scripts/download_polytopia_images.py --delay 1.0
```

Limiter à 20 pages :

```bash
python scripts/download_polytopia_images.py --max-pages 20
```

Télécharger dans un dossier personnalisé :

```bash
python scripts/download_polytopia_images.py --output-dir /path/to/icons
```

## Organisation des images

Les images sont automatiquement organisées dans des sous-dossiers selon leur catégorie :

- `terrain/` : Images de terrain (plaine, forêt, montagne, eau, etc.)
- `units/` : Images d'unités (guerrier, géant, navire, etc.)
- `cities/` : Images de villes et capitales
- `tech/` : Images de technologies
- `tribes/` : Images de tribus/civilisations
- `other/` : Autres images

## Notes importantes

1. **Respect du serveur** : Le script inclut des délais entre les requêtes pour ne pas surcharger le serveur du wiki. Ne réduisez pas trop le délai.

2. **Images déjà téléchargées** : Le script saute automatiquement les images qui existent déjà pour éviter les téléchargements inutiles.

3. **Nettoyage des URLs** : Les URLs sont nettoyées pour enlever les paramètres de redimensionnement et récupérer les images en taille originale.

4. **Catégorisation automatique** : Le script tente de catégoriser automatiquement les images basé sur leur nom et contexte, mais certaines peuvent finir dans `other/`.

## Dépannage

Si le script échoue :

1. Vérifiez votre connexion internet
2. Vérifiez que les dépendances sont installées
3. Augmentez le délai (`--delay 2.0`)
4. Limitez le nombre de pages (`--max-pages 10`)

