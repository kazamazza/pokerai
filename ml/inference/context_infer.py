from typing import List, Set, Optional


class ContextInferer:
    @staticmethod
    def infer_from_request(req: "PolicyRequest") -> Optional[str]:
        try:
            street = int(getattr(req, "street", 0) or 0)
            if street < 1:
                return None  # Only relevant postflop

            hero = getattr(req, "hero_pos", "").upper()
            vill = getattr(req, "villain_pos", "").upper()
            hero_vill = {hero, vill}

            history = req.actions_hist or []
            preflop = history if street == 1 else [a for a in history if int(getattr(a, "street", 0)) == 0]

            preflop_actions = [getattr(a, "action", "").upper() for a in preflop]
            n_raises = preflop_actions.count("RAISE")
            n_calls = preflop_actions.count("CALL")
            n_limps = preflop_actions.count("LIMP")

            def is_sbbb(pos): return pos in {"SB", "BB"}
            def is_btn_co(pos): return pos in {"BTN", "CO"}

            if n_raises == 0:
                if hero_vill == {"SB", "BB"}:
                    return "LIMPED_SINGLE"

            elif n_raises == 1:
                if is_sbbb(hero) and is_btn_co(vill):
                    return "BLIND_VS_STEAL"
                return "SRP_OOP" if is_sbbb(hero) else "SRP_IP"

            elif n_raises == 2:
                return "VS_3BET_OOP" if is_sbbb(hero) else "VS_3BET_IP"

            elif n_raises >= 3:
                return "VS_4BET_OOP" if is_sbbb(hero) else "VS_4BET_IP"

            return None
        except Exception as e:
            print("[predict] failed to infer context:", e)
            return None