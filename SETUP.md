# Guide de configuration

## Environnement virtuel Python

```bash
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
```

## Installation

```bash
pip install -e .
# Optionnel pour les wrappers RL Gymnasium / PettingZoo :
pip install -e .[rl]
# Optionnel pour le développement :
pip install -e .[dev]
```

## Vérification

```bash
python -c "from polytopia_jax.core.init import init_random; print('OK')"
pytest tests/ -v
```
