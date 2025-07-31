from typing import Literal, Dict


class BoardTexture:
    def __init__(
        self,
        structure: Literal["paired", "connected", "uncoordinated"],
        suit_texture: Literal["monotone", "two-tone", "rainbow"],
        suits: Dict[str, int],
        is_paired: bool,
        is_monotone: bool,
        is_two_tone: bool,
        is_connected: bool,
        has_flush_draw: bool,
        has_backdoor_flush_draw: bool,
        has_straight_draw: bool,
        is_flush_possible: bool,
        is_straight_possible: bool,
        high_card_rank: int,
        board_class: str,
        rank_cluster: str,
        has_backdoor_straight_draw: bool,                 # e.g., needs two cards to complete a straight
        texture_blocker_influence_score: float ,           # 0.0–1.0: how sensitive board is to blockers
        coordination_density_score: float,                 # 0.0–1.0: how wet/dynamic the board is
        board_cluster_id: int                             # e.g., ID from BoardClusterer
    ):
        self.structure = structure                              # "paired", "connected", "uncoordinated"
        self.suit_texture = suit_texture                        # "monotone", "two-tone", "rainbow"
        self.suits = suits                                      # e.g., {"h": 2, "s": 1}
        self.is_paired = is_paired
        self.is_monotone = is_monotone
        self.is_two_tone = is_two_tone
        self.is_connected = is_connected
        self.has_flush_draw = has_flush_draw
        self.has_backdoor_flush_draw = has_backdoor_flush_draw
        self.has_straight_draw = has_straight_draw
        self.is_flush_possible = is_flush_possible
        self.is_straight_possible = is_straight_possible
        self.high_card_rank = high_card_rank                    # Integer representation of highest card rank
        self.board_class = board_class                          # e.g., "high_card", "wet", "dry", etc.
        self.rank_cluster = rank_cluster                        # e.g., "low", "mid", "high", "broadway"
        self.has_backdoor_straight_draw = has_backdoor_straight_draw
        self.texture_blocker_influence_score = texture_blocker_influence_score
        self.coordination_density_score = coordination_density_score
        self.board_cluster_id = board_cluster_id

    def __repr__(self):
        return (
            f"<BoardTexture structure={self.structure}, suits={self.suit_texture}, "
            f"class={self.board_class}, straight_draw={self.has_straight_draw}, "
            f"flush_draw={self.has_flush_draw}, high_card={self.high_card_rank}>"
        )

    def to_dict(self) -> dict:
        return {
            "structure": self.structure,
            "suit_texture": self.suit_texture,
            "suits": self.suits,
            "is_paired": self.is_paired,
            "is_monotone": self.is_monotone,
            "is_connected": self.is_connected,
            "has_flush_draw": self.has_flush_draw,
            "has_backdoor_flush_draw": self.has_backdoor_flush_draw,
            "has_straight_draw": self.has_straight_draw,
            "is_flush_possible": self.is_flush_possible,
            "is_straight_possible": self.is_straight_possible,
            "high_card_rank": self.high_card_rank,
            "board_class": self.board_class,
            "rank_cluster": self.rank_cluster,
            "has_backdoor_straight_draw": self.has_backdoor_straight_draw,
            "texture_blocker_influence_score": self.texture_blocker_influence_score,
            "coordination_density_score": self.coordination_density_score,
            "board_cluster_id": self.board_cluster_id,
        }