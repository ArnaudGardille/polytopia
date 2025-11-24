# Guide d'Installation et d'Utilisation - Scraper Wiki Polytopia

## 🚀 Installation Rapide

### Étape 1 : Vérifier Python

```bash
python3 --version
```

Vous devez avoir Python 3.7 ou supérieur.

### Étape 2 : Installer les dépendances

```bash
cd /Users/arnaud/Development/polytopia/scripts
pip3 install -r requirements_scraper.txt
```

### Étape 3 : Test rapide (optionnel mais recommandé)

Testez d'abord sur une seule page :

```bash
python3 test_scraper.py
```

Cela créera un dossier `wiki_test/` avec le résultat du scraping d'une seule page.

### Étape 4 : Lancer le scraping complet

```bash
# Scraping basique (50 pages)
python3 scrape_wiki.py

# Ou avec plus de pages
python3 scrape_wiki.py --max-pages 100
```

## 📋 Commandes Utiles

### Vérifier robots.txt avant de commencer

```bash
python3 scrape_wiki.py --check-robots
```

### Scraping avec paramètres personnalisés

```bash
# 100 pages, délai de 3 secondes, dossier personnalisé
python3 scrape_wiki.py \
  --max-pages 100 \
  --delay 3.0 \
  --output ./knowledge_base
```

### Voir l'aide complète

```bash
python3 scrape_wiki.py --help
```

## 📁 Résultats

Les fichiers seront organisés dans le dossier `wiki_knowledge/` (ou celui spécifié) :

```
wiki_knowledge/
├── images/              # Images téléchargées
├── game_mechanics/      # Mécaniques de jeu
│   └── Map_Generation.md
├── tribes/              # Tribus
│   ├── Xin-xi.md
│   └── Imperius.md
├── units/               # Unités
│   ├── Warrior.md
│   └── Archer.md
└── ...
```

## 🔧 Dépannage

### Erreur : "ModuleNotFoundError"

```bash
pip3 install -r requirements_scraper.txt
```

### Erreur de connexion

- Vérifiez votre connexion Internet
- Essayez d'augmenter le délai : `--delay 5.0`

### Le script est trop lent

C'est normal ! Le rate limiting (2 secondes par page) est intentionnel pour respecter le serveur.

### Permissions refusées

```bash
chmod +x scrape_wiki.py test_scraper.py
```

## ⚠️ Rappels Importants

1. **Licence CC-BY-SA** : Le contenu scrapy doit être attribué
2. **Rate limiting** : Ne réduisez pas le délai en dessous de 1 seconde
3. **Usage responsable** : N'abusez pas du scraping
4. **robots.txt** : Vérifiez les règles avec `--check-robots`

## 💡 Conseils

1. **Commencez petit** : Testez d'abord avec `--max-pages 10`
2. **Vérifiez les résultats** : Regardez les fichiers générés avant de continuer
3. **Sauvegardez régulièrement** : Le scraping peut être interrompu

## 🎯 Utilisation dans l'Application

Une fois le scraping terminé, vous pouvez utiliser les fichiers Markdown comme base de connaissance pour votre application Polytopia.

### Exemple d'intégration

```python
from pathlib import Path

# Charger la base de connaissance
knowledge_dir = Path("wiki_knowledge")

# Lire un fichier spécifique
map_gen_file = knowledge_dir / "game_mechanics" / "Map_Generation.md"
with open(map_gen_file, 'r', encoding='utf-8') as f:
    content = f.read()
    
# Utiliser le contenu dans votre application
# (recherche, affichage, AI training, etc.)
```

## 📞 Support

Pour des questions ou problèmes :

1. Vérifiez le README_SCRAPER.md pour plus de détails
2. Consultez la documentation de Fandom : https://www.fandom.com/terms-of-use
3. Vérifiez que les dépendances sont installées correctement

---

**Bonne chance avec votre base de connaissance Polytopia ! 🎮**


