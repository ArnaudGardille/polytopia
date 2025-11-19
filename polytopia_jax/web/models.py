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
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": 1,
                "pos": [3, 4],
                "hp": 10,
                "owner": 0
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
    current_player: int = Field(..., description="Joueur actif")
    turn: int = Field(..., description="Numéro de tour")
    done: bool = Field(..., description="Partie terminée")
    
    @classmethod
    def from_raw_state(cls, state: dict) -> "GameStateView":
        """Crée un GameStateView depuis un état brut sérialisé."""
        terrain = state["terrain"]
        
        # Extraire les villes depuis city_owner et city_level
        cities = []
        city_owner = state.get("city_owner", [])
        city_level = state.get("city_level", [])
        
        if city_owner and city_level:
            height = len(city_owner)
            width = len(city_owner[0]) if height > 0 else 0
            
            for y in range(height):
                for x in range(width):
                    owner = city_owner[y][x]
                    level = city_level[y][x]
                    if owner != -1 and level > 0:  # -1 signifie pas de ville
                        cities.append(CityView(
                            owner=owner,
                            level=level,
                            pos=[x, y]
                        ))
        
        # Extraire les unités
        units = []
        raw_units = state.get("units", [])
        for idx, unit in enumerate(raw_units):
            units.append(UnitView(
                id=unit.get("id", idx),
                type=unit["type"],
                pos=unit["pos"],
                hp=unit["hp"],
                owner=unit["owner"]
            ))
        
        return cls(
            terrain=terrain,
            cities=cities,
            units=units,
            current_player=state.get("current_player", 0),
            turn=state.get("turn", 0),
            done=state.get("done", False)
        )
    
    class Config:
        json_schema_extra = {
            "example": {
                "terrain": [[0, 1], [1, 0]],
                "cities": [{"owner": 0, "level": 1, "pos": [0, 0]}],
                "units": [{"type": 1, "pos": [1, 1], "hp": 10, "owner": 0}],
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
    difficulty: str = Field("crazy", description="Difficulté symbolique (IA future)")
    seed: Optional[int] = Field(None, description="Seed optionnelle pour reproductibilité")


class LiveActionPayload(BaseModel):
    """Payload d'action encodée pour une partie live."""

    action_id: int = Field(..., ge=0, description="Action encodée (voir core.actions)")


class LiveGameResponse(BaseModel):
    """Réponse standard pour les endpoints live."""

    game_id: str = Field(..., description="Identifiant de la partie live")
    max_turns: int = Field(..., description="Limite de tours pour le mode")
    opponents: int = Field(..., description="Nombre d'adversaires IA")
    difficulty: str = Field(..., description="Difficulté choisie")
    state: GameStateView = Field(..., description="État courant de la partie")

