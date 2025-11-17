# Guide de configuration

## Environnement virtuel Python

Un environnement virtuel Python a été créé dans le dossier `venv/`.

### Activer l'environnement virtuel

**Sur macOS/Linux :**
```bash
source venv/bin/activate
```

**Sur Windows :**
```bash
venv\Scripts\activate
```

Une fois activé, vous verrez `(venv)` au début de votre ligne de commande.

### Désactiver l'environnement virtuel

```bash
deactivate
```

## Installation des dépendances

Les dépendances principales sont déjà installées. Si vous devez réinstaller :

```bash
# Activer l'environnement d'abord
source venv/bin/activate

# Installer les dépendances web (si nécessaire)
pip install 'fastapi>=0.104.0' 'uvicorn[standard]>=0.24.0' 'pydantic>=2.0.0' 'httpx>=0.25.0'

# Installer le projet en mode développement
pip install -e . --no-compile
```

Note : L'option `--no-compile` évite les erreurs de compilation avec JAX sur Python 3.9.

## Lancer le serveur web

```bash
# Activer l'environnement
source venv/bin/activate

# Lancer le serveur
python scripts/run_web_demo.py
```

Ou directement avec uvicorn :

```bash
source venv/bin/activate
uvicorn polytopia_jax.web.api:app --reload --host 0.0.0.0 --port 8000
```

Le serveur sera accessible sur :
- API : http://localhost:8000
- Documentation Swagger : http://localhost:8000/docs
- Documentation ReDoc : http://localhost:8000/redoc

## Exécuter les tests

```bash
source venv/bin/activate
pytest tests/test_web/ -v
```

## Vérifier que tout fonctionne

```bash
source venv/bin/activate
python -c "from polytopia_jax.web.api import app; print('✓ Backend web prêt!')"
```

