from utils.context_validators import valid_open_pair, valid_vs_3bet, valid_vs_4bet, valid_vs_limp, valid_vs_iso, \
    valid_vs_open, valid_multiway, valid_rake_tier, valid_open_policy


def preflop_jobs_for_tuple(
    context: str,
    ip: str,
    oop: str,
    stack: int,
    expl: str,
    mw: str,
    pop: str,
    *,
    rake_tier: str = "MID",
    ante_bb: float = 0.0,
    open_size_policy: str = "STD",
) -> list[dict]:
    """
    Emit jobs for exactly the requested context (or [] if invalid),
    with axis-aware validators (rake/ante/open-policy).
    """
    msgs: list[dict] = []
    if not valid_multiway(mw, stack=stack):
        return msgs
    if not valid_rake_tier(rake_tier):
        return msgs
    if not valid_open_policy(open_size_policy, stack=stack, ip=ip):
        return msgs

    base = dict(
        stack_bb=stack,
        exploit_setting=expl,
        multiway_context=mw,
        population_type=pop,
        rake_tier=rake_tier,
        ante_bb=ante_bb,
        open_size_policy=open_size_policy,
    )

    if context == "OPEN":
        if valid_open_pair(ip, oop):
            msgs.append(dict(base, ip_position=ip, oop_position=oop, action_context="OPEN"))

    elif context == "VS_OPEN":
        # defender (ip) must act after the opener (oop)
        if valid_open_pair(oop, ip) and valid_vs_open(ip, oop):
            msgs.append(dict(base, ip_position=ip, oop_position=oop, action_context="VS_OPEN"))

    elif context == "VS_3BET":
        if valid_vs_3bet(ip, oop):
            msgs.append(dict(base, ip_position=ip, oop_position=oop, action_context="VS_3BET"))

    elif context == "VS_4BET":
        if valid_vs_4bet(ip, oop):
            msgs.append(dict(base, ip_position=ip, oop_position=oop, action_context="VS_4BET"))

    elif context == "VS_LIMP":
        # Only generate limped pots when there’s an ante configured
        if ante_bb > 0 and valid_vs_limp(ip, oop):
            msgs.append(dict(base, ip_position=ip, oop_position=oop, action_context="VS_LIMP"))

    elif context == "VS_ISO":
        # ISO only exists if limps exist in this game config
        if ante_bb > 0 and valid_vs_iso(ip, oop):
            msgs.append(dict(base, ip_position=ip, oop_position=oop, action_context="VS_ISO"))

    return msgs