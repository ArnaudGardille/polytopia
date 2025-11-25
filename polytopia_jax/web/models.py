"""Modèles Pydantic pour la sérialisation des états de jeu."""

from typing import List, Optional
from pydantic import BaseModel, Field


class UnitView(BaseModel):
    """Représentation d'une unité."""
    
    id: int = Field(-1, description="ID interne de l'unité")
    type: int = Field(..., description="Type d'unité")
    pos: List[int] = Field(..., description="Position [x, y]")
    hp: int = Field(..., description="Points de vie")
    owner: int = Field(..., description="Propriétaire de l'unité")
    has_acted: bool = Field(False, description="L'unité a déjà agi ce tour")
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": 1,
                "pos": [3, 4],
                "hp": 10,
                "owner": 0,
                "has_acted": False,
            }
        }


class CityView(BaseModel):
    """Représentation d'une ville."""
    
    owner: int = Field(..., description="Propriétaire de la ville")
    level: int = Field(..., description="Niveau de la ville")
    pos: List[int] = Field(..., description="Position [x, y]")
    
    class Config:
        json_schema_extra = {
            "example": {
                "owner": 0,
                "level": 1,
                "pos": [2, 3]
            }
        }


class GameStateView(BaseModel):
    """Version sérialisée optimisée pour l'affichage."""
    
    terrain: List[List[int]] = Field(..., description="Grille de terrain [H, W]")
    cities: List[CityView] = Field(default_factory=list, description="Liste des villes")
    units: List[UnitView] = Field(default_factory=list, description="Liste des unités")
    city_population: List[List[int]] = Field(default_factory=list, description="Population de chaque case ville")
    city_ports: List[List[int]] = Field(default_factory=list, description="Ports présents sur les villes")
    resource_type: List[List[int]] = Field(default_factory=list, description="Type de ressource par case")
    resource_available: List[List[bool]] = Field(default_factory=list, description="Disponibilité des ressources (True = disponible)")
    player_stars: List[int] = Field(default_factory=list, description="Ressources économiques par joueur")
    player_income: List[int] = Field(default_factory=list, description="Gain d'étoiles par tour pour chaque joueur")
    player_score: List[int] = Field(default_factory=list, description="Score total par joueur")
    score_breakdown: dict = Field(default_factory=dict, description="Breakdown des scores (territoire, population, militaire, économie)")
    player_techs: List[List[int]] = Field(default_factory=list, description="Technologies débloquées par joueur")
    visibility_mask: Optional[List[List[int]]] = Field(
        None,
        description="Masque de visibilité (1 = case révélée)",
    )
    current_player: int = Field(..., description="Joueur actif")
    turn: int = Field(..., description="Numéro de tour")
    done: bool = Field(..., description="Partie terminée")
    
    @classmethod
    def from_raw_state(cls, state: dict) -> "GameStateView":
        """Crée un GameStateView depuis un état brut sérialisé."""
        import time
        start = time.time()
        
        terrain = state["terrain"]
        
        # Récupérer le masque de visibilité
        visibility_mask = state.get("visibility_mask")
        current_player = state.get("current_player", 0)
        
        # Masquer le terrain selon la visibilité
        if visibility_mask:
            height = len(terrain)
            width = len(terrain[0]) if height > 0 else 0
            masked_terrain = [
                [
                    terrain[y][x] if (y < len(visibility_mask) and x < len(visibility_mask[y]) and visibility_mask[y][x] == 1)
                    else 0  # Terrain masqué = 0 (PLAIN)
                    for x in range(width)
                ]
                for y in range(height)
            ]
            terrain = masked_terrain
        
        # Extraire les villes depuis city_owner et city_level (optimisé)
        cities = []
        city_owner = state.get("city_owner", [])
        city_level = state.get("city_level", [])
        city_population = state.get("city_population", [])
        city_ports = state.get("city_has_port", [])
        
        # Récupérer le masque de visibilité
        visibility_mask = state.get("visibility_mask")
        current_player = state.get("current_player", 0)
        
        # Si pas de masque, considérer toute la carte visible
        is_visible = None
        if visibility_mask:
            is_visible = lambda y, x: visibility_mask[y][x] == 1 if y < len(visibility_mask) and x < len(visibility_mask[y]) else False
        else:
            is_visible = lambda y, x: True
        
        if city_owner and city_level:
            height = len(city_owner)
            width = len(city_owner[0]) if height > 0 else 0
            
            # Utiliser une liste en compréhension pour être plus rapide
            # Filtrer selon la visibilité
            cities = [
                CityView(owner=city_owner[y][x], level=city_level[y][x], pos=[x, y])
                for y in range(height)
                for x in range(width)
                if city_owner[y][x] != -1 and city_level[y][x] > 0 and is_visible(y, x)
            ]
        
        # Extraire les unités (optimisé avec liste en compréhension)
        # Filtrer selon la visibilité
        raw_units = state.get("units", [])
        units = [
            UnitView(
                id=unit.get("id", idx),
                type=unit["type"],
                pos=unit["pos"],
                hp=unit["hp"],
                owner=unit["owner"],
                has_acted=unit.get("has_acted", False),
            )
            for idx, unit in enumerate(raw_units)
            if is_visible(unit["pos"][1], unit["pos"][0])  # pos = [x, y], donc y=pos[1], x=pos[0]
        ]
        
        player_stars = state.get("player_stars", [])
        player_income = state.get("player_income", [])
        player_score = state.get("player_score", [])
        score_breakdown = state.get("score_breakdown", {})
        player_techs = state.get("player_techs", [])
        resource_type = state.get("resource_type", [])
        resource_available = state.get("resource_available", [])
        visibility_mask = state.get("visibility_mask")
        
        # Utiliser model_validate pour éviter la validation supplémentaire
        result = cls.model_validate({
            "terrain": terrain,
            "cities": cities,
            "units": units,
            "city_population": city_population,
            "city_ports": city_ports,
            "resource_type": resource_type,
            "resource_available": resource_available,
            "player_stars": player_stars,
            "player_income": player_income,
            "player_score": player_score,
            "score_breakdown": score_breakdown,
            "player_techs": player_techs,
            "visibility_mask": visibility_mask,
            "current_player": state.get("current_player", 0),
            "turn": state.get("turn", 0),
            "done": state.get("done", False)
        })
        
        return result
    
    class Config:
        json_schema_extra = {
            "example": {
                "terrain": [[0, 1], [1, 0]],
                "cities": [{"owner": 0, "level": 1, "pos": [0, 0]}],
                "units": [{"type": 1, "pos": [1, 1], "hp": 10, "owner": 0}],
                "city_population": [[0, 0], [0, 1]],
                "city_ports": [[0, 0], [0, 1]],
                "player_stars": [5, 5],
                "player_income": [3, 2],
                "player_score": [120, 80],
                "score_breakdown": {
                    "territory": [100, 0],
                    "population": [10, 5],
                    "military": [10, 10],
                    "economy": [0, 65],
                },
                "player_techs": [[1, 0, 0, 0], [1, 0, 0, 0]],
                "visibility_mask": [[1, 1], [1, 1]],
                "current_player": 0,
                "turn": 1,
                "done": False
            }
        }


class ReplayMetadata(BaseModel):
    """Métadonnées d'un replay."""
    
    height: int = Field(..., description="Hauteur de la carte")
    width: int = Field(..., description="Largeur de la carte")
    num_players: int = Field(..., description="Nombre de joueurs")
    max_turns: Optional[int] = Field(None, description="Limite de tours")
    seed: Optional[int] = Field(None, description="Seed aléatoire")
    final_turn: Optional[int] = Field(None, description="Dernier tour")
    total_actions: Optional[int] = Field(None, description="Nombre total d'actions")
    game_done: Optional[bool] = Field(None, description="Partie terminée")
    
    class Config:
        json_schema_extra = {
            "example": {
                "height": 10,
                "width": 10,
                "num_players": 2,
                "max_turns": 100,
                "seed": 42,
                "final_turn": 50,
                "total_actions": 200,
                "game_done": False
            }
        }


class GameInfo(BaseModel):
    """Info d'un replay pour la liste."""
    
    id: str = Field(..., description="Identifiant du replay")
    metadata: ReplayMetadata = Field(..., description="Métadonnées du replay")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "test_game",
                "metadata": {
                    "height": 10,
                    "width": 10,
                    "num_players": 2
                }
            }
        }


class ReplayResponse(BaseModel):
    """Réponse complète d'un replay."""
    
    id: str = Field(..., description="Identifiant du replay")
    metadata: ReplayMetadata = Field(..., description="Métadonnées du replay")
    states: List[GameStateView] = Field(..., description="Liste des états du jeu")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "test_game",
                "metadata": {
                    "height": 10,
                    "width": 10,
                    "num_players": 2
                },
                "states": []
            }
        }


class StateResponse(BaseModel):
    """Réponse pour un état à un tour donné."""
    
    turn: int = Field(..., description="Numéro de tour")
    state: GameStateView = Field(..., description="État du jeu à ce tour")
    
    class Config:
        json_schema_extra = {
            "example": {
                "turn": 5,
                "state": {
                    "terrain": [[0, 1], [1, 0]],
                    "cities": [],
                    "units": [],
                    "current_player": 0,
                    "turn": 5,
                    "done": False
                }
            }
        }


class GamesListResponse(BaseModel):
    """Réponse pour la liste des replays."""
    
    games: List[GameInfo] = Field(..., description="Liste des replays disponibles")
    
    class Config:
        json_schema_extra = {
            "example": {
                "games": [
                    {
                        "id": "test_game",
                        "metadata": {
                            "height": 10,
                            "width": 10,
                            "num_players": 2
                        }
                    }
                ]
            }
        }


class LiveGameConfig(BaseModel):
    """Payload de configuration pour une partie live."""

    opponents: int = Field(3, ge=1, le=9, description="Nombre d'adversaires IA")
    difficulty: str = Field("crazy", description="Difficulté (impact les bonus d'étoiles des IA)")
    strategy: Optional[str] = Field(
        None,
        description="Stratégie IA (idle, random, rush, economy)",
    )
    seed: Optional[int] = Field(None, description="Seed optionnelle pour reproductibilité")
    reveal_map: Optional[bool] = Field(
        None,
        description="Forcer l'affichage de toute la carte",
    )
    unlock_all_techs: Optional[bool] = Field(
        None,
        description="Débloquer automatiquement toutes les technologies",
    )


class LiveActionPayload(BaseModel):
    """Payload d'action encodée pour une partie live."""

    action_id: int = Field(..., ge=0, description="Action encodée (voir core.actions)")


class LiveGameResponse(BaseModel):
    """Réponse standard pour les endpoints live."""

    game_id: str = Field(..., description="Identifiant de la partie live")
    max_turns: int = Field(..., description="Limite de tours pour le mode")
    opponents: int = Field(..., description="Nombre d'adversaires IA")
    difficulty: str = Field(..., description="Difficulté choisie")
    strategy: str = Field(..., description="Stratégie IA utilisée par défaut")
    state: GameStateView = Field(..., description="État courant de la partie")
