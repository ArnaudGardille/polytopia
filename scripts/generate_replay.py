#!/usr/bin/env python3
"""Script pour générer des replays de parties complètes."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp

from polytopia_jax.core.init import init_random, GameConfig
from polytopia_jax.core.rules import step
from polytopia_jax.core.state import GameState

from scripts.serialize import state_to_dict
from scripts.simple_bot import SimpleBot


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Génère un replay de partie complète (bot vs bot)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Chemin du fichier de sortie (défaut: replays/game_{timestamp}.json)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=10,
        help="Hauteur de la carte (défaut: 10)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=10,
        help="Largeur de la carte (défaut: 10)"
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Nombre de joueurs (défaut: 2)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=100,
        help="Limite de tours (défaut: 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed aléatoire pour reproductibilité"
    )
    
    args = parser.parse_args()
    
    # Déterminer le chemin de sortie
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("replays")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"game_{timestamp}.json"
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialiser la seed
    if args.seed is not None:
        key = jax.random.PRNGKey(args.seed)
    else:
        key = jax.random.PRNGKey(42)  # Seed par défaut
    
    # Configuration du jeu
    config = GameConfig(
        height=args.height,
        width=args.width,
        num_players=args.num_players,
        max_units=50,
    )
    
    print(f"Initialisation d'une partie {args.width}x{args.height} avec {args.num_players} joueurs...")
    
    # Initialiser l'état
    state = init_random(key, config)
    
    # Créer les bots
    bots = []
    for player_id in range(args.num_players):
        bot_seed = args.seed + player_id if args.seed is not None else None
        bots.append(SimpleBot(player_id, seed=bot_seed))
    
    # Liste pour stocker les états
    states = []
    
    # Sérialiser l'état initial
    states.append(state_to_dict(state))
    
    print(f"État initial sérialisé. Début de la simulation...")
    
    # Boucle principale de simulation
    turn_count = 0
    actions_count = 0
    
    while not bool(jnp.asarray(state.done)) and turn_count < args.max_turns:
        current_player = int(jnp.asarray(state.current_player))
        current_turn = int(jnp.asarray(state.turn))
        
        # Choisir une action avec le bot du joueur actif
        bot = bots[current_player]
        action = bot.choose_action(state)
        
        # Appliquer l'action
        state = step(state, action)
        actions_count += 1
        
        # Sérialiser l'état après l'action
        states.append(state_to_dict(state))
        
        # Vérifier si on a changé de tour (quand on revient au joueur 0)
        new_turn = int(jnp.asarray(state.turn))
        if new_turn > current_turn:
            turn_count = new_turn
            print(f"Tour {turn_count} terminé (actions: {actions_count})")
        
        # Vérifier si la partie est terminée
        if bool(jnp.asarray(state.done)):
            print(f"Partie terminée au tour {turn_count}!")
            break
    
    if turn_count >= args.max_turns:
        print(f"Limite de tours ({args.max_turns}) atteinte.")
    
    # Préparer les métadonnées
    final_turn = int(jnp.asarray(state.turn))
    metadata = {
        "height": args.height,
        "width": args.width,
        "num_players": args.num_players,
        "max_turns": args.max_turns,
        "seed": args.seed if args.seed is not None else 42,
        "final_turn": final_turn,
        "total_actions": actions_count,
        "game_done": bool(jnp.asarray(state.done)),
    }
    
    # Créer le replay complet
    replay = {
        "metadata": metadata,
        "states": states,
    }
    
    # Sauvegarder en JSON
    print(f"Sauvegarde du replay dans {output_path}...")
    
    # Fonction helper pour convertir les ArrayImpl en types Python natifs
    def convert_to_native(obj):
        """Convertit récursivement les ArrayImpl JAX en types Python natifs."""
        import numpy as np
        if hasattr(obj, 'item'):  # ArrayImpl ou array numpy
            return obj.item()
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        else:
            return obj
    
    # Convertir le replay en types Python natifs
    replay_native = convert_to_native(replay)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(replay_native, f, indent=2, ensure_ascii=False)
    
    print(f"Replay sauvegardé avec succès!")
    print(f"  - {len(states)} états enregistrés")
    print(f"  - {actions_count} actions effectuées")
    print(f"  - Partie terminée: {metadata['game_done']}")


if __name__ == "__main__":
    main()

