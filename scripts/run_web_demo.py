#!/usr/bin/env python3
"""Script pour lancer le serveur web FastAPI."""

import signal
import sys
import uvicorn


def signal_handler(sig, frame):
    """Gère proprement l'arrêt du serveur avec Ctrl+C."""
    print('\n⚠️  Signal d\'arrêt reçu (Ctrl+C)...')
    print('🛑 Arrêt du serveur en cours...')
    sys.exit(0)


if __name__ == "__main__":
    # Enregistrer le gestionnaire de signal pour Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 Démarrage du serveur Polytopia-JAX...")
    print("📍 API accessible sur : http://localhost:8000")
    print("📖 Documentation Swagger : http://localhost:8000/docs")
    print("⚠️  Appuyez sur Ctrl+C pour arrêter le serveur")
    print("-" * 60)
    
    try:
        uvicorn.run(
            "polytopia_jax.web.api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print('\n🛑 Arrêt du serveur...')
        sys.exit(0)

