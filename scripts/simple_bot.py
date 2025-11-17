"""Bot simple pour simulation de parties."""

import random
from typing import Optional
import jax.numpy as jnp
from polytopia_jax.core.state import GameState, NO_OWNER
from polytopia_jax.core.actions import (
    ActionType,
    Direction,
    encode_action,
    END_TURN_ACTION,
)
from polytopia_jax.core.rules import legal_actions_mask


class SimpleBot:
    """Bot simple qui choisit des actions basiques selon des priorités."""
    
    def __init__(self, player_id: int, seed: int = None):
        """Initialise le bot.
        
        Args:
            player_id: ID du joueur contrôlé par ce bot
            seed: Seed pour la génération aléatoire (optionnel)
        """
        self.player_id = player_id
        if seed is not None:
            random.seed(seed)
    
    def choose_action(self, state: GameState) -> int:
        """Choisit une action selon une logique de priorité.
        
        Priorités:
        1. Attaquer une unité ennemie adjacente
        2. Se déplacer vers une ville ennemie ou neutre
        3. Se déplacer aléatoirement
        4. Terminer le tour
        
        Args:
            state: État actuel du jeu
        
        Returns:
            Action encodée
        """
        # Vérifier que c'est bien le tour de ce bot
        current_player = int(jnp.asarray(state.current_player))
        if current_player != self.player_id:
            return END_TURN_ACTION
        
        # Trouver toutes les unités du joueur actif
        player_units = self._get_player_units(state)
        
        if len(player_units) == 0:
            return END_TURN_ACTION
        
        # Priorité 1 : Attaquer une unité ennemie adjacente
        for unit_id in player_units:
            attack_action = self._try_attack(state, unit_id)
            if attack_action is not None:
                return attack_action
        
        # Priorité 2 : Se déplacer vers une ville ennemie ou neutre
        for unit_id in player_units:
            move_to_city_action = self._try_move_to_city(state, unit_id)
            if move_to_city_action is not None:
                return move_to_city_action
        
        # Priorité 3 : Se déplacer aléatoirement
        for unit_id in player_units:
            random_move_action = self._try_random_move(state, unit_id)
            if random_move_action is not None:
                return random_move_action
        
        # Priorité 4 : Terminer le tour
        return END_TURN_ACTION
    
    def _get_player_units(self, state: GameState) -> list:
        """Retourne la liste des IDs d'unités du joueur actif."""
        units = []
        for i in range(state.max_units):
            if (state.units_active[i] and 
                int(jnp.asarray(state.units_owner[i])) == self.player_id):
                units.append(i)
        return units
    
    def _try_attack(self, state: GameState, unit_id: int) -> Optional[int]:
        """Essaie de trouver une attaque valide pour l'unité.
        
        Returns:
            Action d'attaque encodée, ou None si aucune attaque possible
        """
        unit_pos = state.units_pos[unit_id]
        unit_x = int(jnp.asarray(unit_pos[0]))
        unit_y = int(jnp.asarray(unit_pos[1]))
        
        # Vérifier les cases adjacentes pour des unités ennemies
        directions = [
            (0, -1),  # UP
            (1, 0),   # RIGHT
            (0, 1),   # DOWN
            (-1, 0),  # LEFT
        ]
        
        for dx, dy in directions:
            target_x = unit_x + dx
            target_y = unit_y + dy
            
            # Vérifier les limites
            if (target_x < 0 or target_x >= state.width or
                target_y < 0 or target_y >= state.height):
                continue
            
            # Vérifier s'il y a une unité ennemie à cette position
            target_unit_id = self._get_unit_at_position(state, target_x, target_y)
            if target_unit_id >= 0:
                target_owner = int(jnp.asarray(state.units_owner[target_unit_id]))
                if target_owner != self.player_id:
                    # Attaquer cette unité
                    return encode_action(
                        ActionType.ATTACK,
                        unit_id=unit_id,
                        target_pos=(target_x, target_y)
                    )
        
        return None
    
    def _try_move_to_city(self, state: GameState, unit_id: int) -> Optional[int]:
        """Essaie de se déplacer vers une ville ennemie ou neutre.
        
        Returns:
            Action de mouvement encodée, ou None si aucun mouvement possible
        """
        unit_pos = state.units_pos[unit_id]
        unit_x = int(jnp.asarray(unit_pos[0]))
        unit_y = int(jnp.asarray(unit_pos[1]))
        
        # Chercher les villes ennemies ou neutres dans un rayon de 3 cases
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                
                target_x = unit_x + dx
                target_y = unit_y + dy
                
                # Vérifier les limites
                if (target_x < 0 or target_x >= state.width or
                    target_y < 0 or target_y >= state.height):
                    continue
                
                # Vérifier s'il y a une ville
                city_level = int(jnp.asarray(state.city_level[target_y, target_x]))
                if city_level > 0:
                    city_owner = int(jnp.asarray(state.city_owner[target_y, target_x]))
                    if city_owner != self.player_id:  # Ennemie ou neutre
                        # Essayer de se déplacer dans cette direction
                        direction = self._get_direction_towards(
                            unit_x, unit_y, target_x, target_y
                        )
                        if direction is not None:
                            return encode_action(
                                ActionType.MOVE,
                                unit_id=unit_id,
                                direction=direction
                            )
        
        return None
    
    def _try_random_move(self, state: GameState, unit_id: int) -> Optional[int]:
        """Essaie un mouvement aléatoire dans une direction valide.
        
        Returns:
            Action de mouvement encodée, ou None si aucun mouvement possible
        """
        directions = list(Direction)
        random.shuffle(directions)
        
        for direction in directions:
            if direction == Direction.NUM_DIRECTIONS:
                continue
            
            # Vérifier si le mouvement est valide (dans les limites)
            unit_pos = state.units_pos[unit_id]
            unit_x = int(jnp.asarray(unit_pos[0]))
            unit_y = int(jnp.asarray(unit_pos[1]))
            
            delta = self._get_direction_delta(direction)
            new_x = unit_x + delta[0]
            new_y = unit_y + delta[1]
            
            # Vérifier les limites
            if (new_x >= 0 and new_x < state.width and
                new_y >= 0 and new_y < state.height):
                # Vérifier que la case n'est pas occupée
                if not self._is_position_occupied(state, new_x, new_y, unit_id):
                    return encode_action(
                        ActionType.MOVE,
                        unit_id=unit_id,
                        direction=direction
                    )
        
        return None
    
    def _get_unit_at_position(self, state: GameState, x: int, y: int) -> int:
        """Retourne l'ID de l'unité à une position, ou -1 si aucune."""
        for i in range(state.max_units):
            if state.units_active[i]:
                pos = state.units_pos[i]
                pos_x = int(jnp.asarray(pos[0]))
                pos_y = int(jnp.asarray(pos[1]))
                if pos_x == x and pos_y == y:
                    return i
        return -1
    
    def _is_position_occupied(self, state: GameState, x: int, y: int, exclude_unit_id: int = -1) -> bool:
        """Vérifie si une position est occupée par une unité."""
        for i in range(state.max_units):
            if i == exclude_unit_id:
                continue
            if state.units_active[i]:
                pos = state.units_pos[i]
                pos_x = int(jnp.asarray(pos[0]))
                pos_y = int(jnp.asarray(pos[1]))
                if pos_x == x and pos_y == y:
                    return True
        return False
    
    def _get_direction_towards(self, from_x: int, from_y: int, to_x: int, to_y: int) -> Optional[Direction]:
        """Retourne la direction pour aller de (from_x, from_y) vers (to_x, to_y).
        
        Retourne la direction cardinale la plus proche.
        """
        dx = to_x - from_x
        dy = to_y - from_y
        
        # Prioriser la direction avec le plus grand déplacement
        if abs(dx) > abs(dy):
            if dx > 0:
                return Direction.RIGHT
            elif dx < 0:
                return Direction.LEFT
        else:
            if dy > 0:
                return Direction.DOWN
            elif dy < 0:
                return Direction.UP
        
        return None
    
    def _get_direction_delta(self, direction: Direction) -> tuple[int, int]:
        """Retourne le delta (dx, dy) pour une direction."""
        deltas = {
            Direction.UP: (0, -1),
            Direction.RIGHT: (1, 0),
            Direction.DOWN: (0, 1),
            Direction.LEFT: (-1, 0),
        }
        return deltas.get(direction, (0, 0))

