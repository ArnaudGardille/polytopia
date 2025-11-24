# Script de Scraping du Wiki Polytopia

Ce script permet de scraper le contenu du wiki Polytopia et de créer une base de connaissance structurée en fichiers Markdown.

## ⚠️ Important - Considérations Légales

- Le contenu du wiki Polytopia est sous licence **CC-BY-SA**
- Vous devez **attribuer la source** lors de l'utilisation du contenu
- Respectez le fichier `robots.txt` du site
- Utilisez ce script de manière responsable avec un rate limiting approprié
- Le scraping intensif peut surcharger les serveurs - soyez respectueux

## 📋 Prérequis

### Installation des dépendances

```bash
pip install -r requirements_scraper.txt
```

Les bibliothèques nécessaires sont :
- `requests` : pour les requêtes HTTP
- `beautifulsoup4` : pour parser le HTML
- `html2text` : pour convertir HTML en Markdown
- `lxml` : parser HTML performant

## 🚀 Utilisation

### Utilisation basique

```bash
python scrape_wiki.py
```

Cela scrappe jusqu'à 50 pages par défaut dans le dossier `wiki_knowledge/`.

### Options disponibles

```bash
# Vérifier robots.txt avant de commencer
python scrape_wiki.py --check-robots

# Spécifier le nombre de pages à scraper
python scrape_wiki.py --max-pages 20

# Changer le dossier de sortie
python scrape_wiki.py --output ./knowledge_base

# Ajuster le délai entre requêtes (en secondes)
python scrape_wiki.py --delay 3.0

# Combiner plusieurs options
python scrape_wiki.py --max-pages 100 --delay 2.5 --output ./wiki_data
```

### Aide complète

```bash
python scrape_wiki.py --help
```

## 📁 Structure de sortie

Le script organise automatiquement les fichiers par catégories :

```
wiki_knowledge/
├── images/              # Toutes les images téléchargées
├── game_mechanics/      # Mécaniques de jeu (combat, mouvement, etc.)
├── tribes/              # Pages des différentes tribus
├── units/               # Pages des unités
├── technology/          # Pages des technologies
├── buildings/           # Pages des bâtiments
├── city/                # Pages liées aux cités
└── general/             # Autres pages
```

Chaque fichier Markdown contient :
- Le titre de la page
- L'URL source
- La date de scraping
- Le contenu converti en Markdown
- Les images locales (téléchargées)
- Les tableaux convertis en format Markdown

## 📝 Format des fichiers générés

Exemple de fichier généré :

```markdown
# Map Generation

**Source:** https://polytopia.fandom.com/wiki/Map_Generation
**Licence:** CC-BY-SA
**Date de scraping:** 2025-11-20

---

[Contenu de la page en Markdown]

---

*Ce contenu est extrait du wiki Polytopia et est sous licence CC-BY-SA.*
```

## 🎯 Pages scrapées

Le script scrappe automatiquement les pages principales suivantes :

### Game Mechanics
- Map Generation
- Combat
- Movement
- Terrain
- Stars, Score, Ruins
- Game Modes

### Tribes
- Tribus gratuites : Xin-xi, Imperius, Bardur, Oumaji
- Tribus régulières : Kickoo, Hoodrick, Luxidoor, Vengir, Zebasi, Ai-Mo, Quetzali

### Units
- Warrior, Archer, Defender, Rider
- Swordsman, Knight, Giant

### Technology & Buildings
- Technologies principales
- Bâtiments (Bridge, Embassy, Temples)

### City
- City, Population, City Connection

## ⚙️ Fonctionnalités

✅ **Rate limiting** : Délai configurable entre chaque requête (défaut: 2s)  
✅ **Vérification robots.txt** : Option pour vérifier les règles de scraping  
✅ **Organisation automatique** : Fichiers organisés par catégories  
✅ **Téléchargement d'images** : Images sauvegardées localement  
✅ **Conversion de tableaux** : Tableaux HTML → Markdown  
✅ **Métadonnées** : Attribution et source incluses dans chaque fichier  
✅ **Gestion d'erreurs** : Continue même si certaines pages échouent  

## 🔧 Personnalisation

Pour ajouter d'autres pages à scraper, modifiez la liste `main_pages` dans la méthode `scrape_from_sitemap()` du fichier `scrape_wiki.py`.

## 🐛 Dépannage

### Erreur de connexion
- Vérifiez votre connexion Internet
- Le site peut être temporairement indisponible
- Augmentez le délai avec `--delay`

### Pages manquantes
- Certaines pages peuvent avoir une structure différente
- Augmentez `--max-pages` si nécessaire

### Erreurs de parsing
- Certains tableaux complexes peuvent ne pas être parfaitement convertis
- Les images peuvent échouer si elles ne sont plus disponibles

## 📄 Licence

Ce script est fourni à des fins éducatives et de recherche. Le contenu scrapy est sous licence CC-BY-SA et appartient aux contributeurs du wiki Polytopia.

## 🤝 Bonnes pratiques

1. **Ne pas abuser** : Limitez le nombre de requêtes
2. **Respecter le délai** : Utilisez un délai d'au moins 2 secondes
3. **Attribuer la source** : Toujours mentionner la source originale
4. **Usage responsable** : Utilisez les données de manière éthique
5. **Vérifier robots.txt** : Assurez-vous que le scraping est autorisé

---

Pour toute question ou problème, référez-vous à la documentation de Fandom ou contactez les maintenteurs du wiki.


