from typing import List, Any, Dict

from features.board_analyzer import BoardAnalyzer
from features.types import Action, ACTION_TO_INDEX, ACTION_TYPES, RangeFeatureRequest
from utils.range_utils import BUCKET_EDGES, _bucket_index, compute_position_onehot, compute_supplemental_stats, \
    compute_situational_stats, compute_profile_stats


class RangeFeatureExtractor:
    """
    Unified extractor for all Range-Net features: foundational, situational, and supplemental.
    """
    def __init__(self, history: List[Any], board_analyzer: BoardAnalyzer):
        # pre-compute per-player profile stats once
        self.history = history
        self.profile_stats  = compute_profile_stats(history)
        self.config         = RangeFeatureConfig()
        self.board_analyzer = board_analyzer

    def _encode_last_actions(self,
                             actions: List[Action],
                             player_id: str,
                             window: int = 4) -> List[int]:
        """
        Turn the last `window` actions of player_id into a one‐hot vector
        of length window * len(ACTION_TYPES), where each chunk of
        len(ACTION_TYPES) is a one‐hot of that action’s enum position.
        """
        # grab up to the last `window` actions for this player
        recent = [a for a in actions if a.player_id == player_id][-window:]
        vec: List[int] = []

        for act in recent:
            # look up the enum directly, not its .value
            idx = ACTION_TO_INDEX[act.type]  # act.type is an ActionType
            onehot = [0] * len(ACTION_TYPES)
            onehot[idx] = 1
            vec.extend(onehot)

        # pad with zeros if fewer than window actions
        needed = window * len(ACTION_TYPES) - len(vec)
        if needed > 0:
            vec.extend([0] * needed)

        return vec

    def extract_foundational(self, req: RangeFeatureRequest) -> Dict[str, Any]:
        hand = req.current_hand
        pid = req.player_id
        feats: Dict[str, Any] = {}
        stats = self.profile_stats.get(pid, {})

        # 1) Bucketed foundational metrics → one-hot bins
        for metric, edges in BUCKET_EDGES.items():
            # metric keys like 'vpip_pct', 'pfr_pct', etc.
            # compute raw percentage or default to 0.0
            raw = stats.get(metric, 0.0)
            # find bin index
            idx = _bucket_index(raw, edges)
            # one-hot of length (len(edges)+1)
            onehot = [0.0] * (len(edges) + 1)
            onehot[idx] = 1.0
            # add to feats as metric_bin_0, metric_bin_1, ...
            base = metric.replace('_pct', '')
            for i, v in enumerate(onehot):
                feats[f"{base}_bin_{i}"] = v

        # 2) Position one-hot
        if 'position_onehot' in self.config.foundation:
            seat_dicts = [s.dict() for s in hand.seats]
            feats['position_onehot'] = compute_position_onehot(
                seat_dicts, hand.button_seat, pid
            )

        # 3) Last-4 actions one-hot
        if 'last4_actions_onehot' in self.config.foundation:
            feats['last4_actions_onehot'] = self._encode_last_actions(
                hand.actions, pid
            )

        # 4) Board texture flags
        if any(key.startswith('board_') for key in self.config.foundation):
            tex = self.board_analyzer.analyze(hand.board)
            feats['board_paired'] = float(tex.is_paired)
            feats['board_connected'] = float(tex.is_connected)
            feats['board_uncoordinated'] = float(not (tex.is_paired or tex.is_connected))
            if not tex.is_paired:
                feats['board_monotone'] = float(tex.is_monotone)
                feats['board_two_tone'] = float(tex.is_two_tone)
                feats['board_rainbow'] = float(not (tex.is_monotone or tex.is_two_tone))
            else:
                feats['board_monotone'] = 0.0
                feats['board_two_tone'] = 0.0
                feats['board_rainbow'] = 0.0

        return feats

    def extract_situational(self, req: RangeFeatureRequest) -> Dict[str, Any]:
        """
        Collect all situational features via the helper in range_utils.
        """
        # Pull in the full suite of situational metrics
        situational_feats = compute_situational_stats(req, self.profile_stats)
        return situational_feats

    def extract_supplemental(self, req: RangeFeatureRequest) -> Dict[str, Any]:
        """
        Collect all supplemental features via the helper in range_utils.
        """
        # Delegate to the shared utility that buckets & one-hots everything
        return compute_supplemental_stats(req, self.profile_stats)

    def extract(self, req: RangeFeatureRequest) -> Dict[str, Any]:
        feats = {}
        feats.update(self.extract_foundational(req))
        feats.update(self.extract_situational(req))
        feats.update(self.extract_supplemental(req))
        return feats