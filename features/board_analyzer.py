from collections import Counter
from typing import List, cast, Literal

from features.board_clusterer import BoardClusterer
from features.board_texture import BoardTexture


class BoardAnalyzer:

    def __init__(self, board_clusterer: BoardClusterer):
        self.board_clusterer = board_clusterer

    def analyze(self, board: List[str]) -> BoardTexture:
        if not board:
            return self._empty_texture()

        ranks = [card[:-1] for card in board]
        suits = [card[-1] for card in board]
        rank_values = sorted([self._rank_to_int(r) for r in ranks])
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)

        is_paired = any(count >= 2 for count in rank_counts.values())
        is_connected = self._is_connected(rank_values)
        structure = "paired" if is_paired else "connected" if is_connected else "uncoordinated"

        num_suits = len(suit_counts)
        is_monotone = (num_suits == 1)
        is_two_tone = (num_suits == 2)
        # anything else (3, 4 or 5 suits) is rainbow
        suit_texture = (
            "monotone" if is_monotone
            else "two-tone" if is_two_tone
            else "rainbow"
        )

        has_flush_draw = any(count == 3 for count in suit_counts.values()) and len(board) == 3
        has_backdoor_flush_draw = any(count == 2 for count in suit_counts.values()) and len(board) >= 3
        is_flush_possible = any(count >= 5 for count in suit_counts.values())
        is_straight_possible = self._has_straight(rank_values)
        has_straight_draw = self._has_straight_draw(rank_values)
        has_backdoor_straight_draw = self._has_backdoor_straight_draw(rank_values)
        high_card_rank = max(rank_values)

        rank_cluster = self._rank_cluster(rank_values)
        board_class = self._classify_board(high_card_rank, structure, suit_texture)

        texture_blocker_influence_score = (
            0.8 if is_connected else 0.5 if is_paired else 0.3
        )
        if is_monotone:
            texture_blocker_influence_score += 0.2
        elif is_two_tone:
            texture_blocker_influence_score += 0.1
        texture_blocker_influence_score = min(texture_blocker_influence_score, 1.0)

        coordination_density_score = 0.0
        if is_connected:
            coordination_density_score += 0.4
        if has_flush_draw or has_straight_draw:
            coordination_density_score += 0.3
        if is_paired:
            coordination_density_score += 0.2
        coordination_density_score = min(coordination_density_score, 1.0)

        board_cluster_id = self._lookup_board_cluster_id(board)

        structure_final = cast(Literal["paired", "connected", "uncoordinated"], structure)
        suit_texture_final = cast(Literal["monotone", "two-tone", "rainbow"], suit_texture)

        return BoardTexture(
            structure=structure_final,
            suit_texture=suit_texture_final,
            suits=dict(suit_counts),
            is_paired=is_paired,
            is_monotone=is_monotone,
            is_two_tone=is_two_tone,
            is_connected=is_connected,
            has_flush_draw=has_flush_draw,
            has_backdoor_flush_draw=has_backdoor_flush_draw,
            is_flush_possible=is_flush_possible,
            is_straight_possible=is_straight_possible,
            has_straight_draw=has_straight_draw,
            has_backdoor_straight_draw=has_backdoor_straight_draw,
            high_card_rank=high_card_rank,
            board_class=board_class,
            rank_cluster=rank_cluster,
            texture_blocker_influence_score=texture_blocker_influence_score,
            coordination_density_score=coordination_density_score,
            board_cluster_id=board_cluster_id
        )

    def _rank_to_int(self, rank: str) -> int:
        return self.RANK_ORDER.get(rank.upper(), 0)

    def _is_connected(self, ranks: List[int]) -> bool:
        return len(ranks) >= 2 and max(ranks) - min(ranks) <= 4 and len(set(ranks)) == len(ranks)

    def _has_straight_draw(self, ranks: List[int]) -> bool:
        uniq = sorted(set(ranks))
        for i in range(len(uniq) - 2):
            if uniq[i+2] - uniq[i] <= 4:
                return True
        return False

    def _has_backdoor_straight_draw(self, ranks: List[int]) -> bool:
        uniq = sorted(set(ranks))
        for i in range(len(uniq) - 1):
            if uniq[i+1] - uniq[i] == 2:
                return True
        return False

    def _has_straight(self, ranks: List[int]) -> bool:
        uniq = sorted(set(ranks))
        for i in range(len(uniq) - 4):
            if uniq[i+4] - uniq[i] == 4:
                return True
        return False

    def _rank_cluster(self, ranks: List[int]) -> str:
        if all(r <= 8 for r in ranks):
            return "low"
        elif all(r >= 9 for r in ranks):
            return "high"
        elif all(6 <= r <= 11 for r in ranks):
            return "mid"
        return "mixed"

    def _classify_board(self, high_rank: int, structure: str, suit_texture: str) -> str:
        if high_rank >= 13 and structure == "uncoordinated" and suit_texture == "rainbow":
            return "Ace-high dry"
        if structure == "paired" and suit_texture != "monotone":
            return "Paired semi-wet"
        if structure == "connected" and suit_texture == "two-tone":
            return "Draw-heavy"
        return "Generic"

    def _lookup_board_cluster_id(self, board: List[str]) -> int:
        return  self.board_clusterer.get_cluster_id(board)

    def _empty_texture(self) -> BoardTexture:
        return BoardTexture(
            structure="uncoordinated",
            suit_texture="rainbow",
            suits={},
            is_paired=False,
            is_monotone=False,
            is_two_tone=False,
            is_connected=False,
            has_flush_draw=False,
            has_backdoor_flush_draw=False,
            is_flush_possible=False,
            is_straight_possible=False,
            has_straight_draw=False,
            has_backdoor_straight_draw=False,
            high_card_rank=0,
            board_class="empty",
            rank_cluster="low",
            texture_blocker_influence_score=0.0,
            coordination_density_score=0.0,
            board_cluster_id=0
        )

    def get_cluster_id(self, board: List[str]) -> str:
        return self.board_clusterer.get_cluster_key(board)