from utils.builders import build_preflop_s3_key_components

# S3 key builders for dependency checks (mirror what the extractor needs)
def deps_keys_for_open(ip, oop, stack, prof, expl, mw, pop):
    return [
        build_preflop_s3_key_components(ip=ip,  oop=oop, stack_bb=stack,
                                        villain_profile=prof, exploit_setting=expl,
                                        multiway_context=mw, population_type=pop,
                                        action_context="OPEN"),
        build_preflop_s3_key_components(ip=oop, oop=ip,  stack_bb=stack,
                                        villain_profile=prof, exploit_setting=expl,
                                        multiway_context=mw, population_type=pop,
                                        action_context="VS_OPEN"),
    ]

def deps_keys_for_vs_3bet(ip, oop, stack, prof, expl, mw, pop):
    # Keep it symmetric for robustness (both POVs present)
    return [
        build_preflop_s3_key_components(ip=ip,  oop=oop, stack_bb=stack,
                                        villain_profile=prof, exploit_setting=expl,
                                        multiway_context=mw, population_type=pop,
                                        action_context="VS_3BET"),
        build_preflop_s3_key_components(ip=oop, oop=ip,  stack_bb=stack,
                                        villain_profile=prof, exploit_setting=expl,
                                        multiway_context=mw, population_type=pop,
                                        action_context="VS_3BET"),
    ]

def deps_keys_for_vs_limp(ip, oop, stack, prof, expl, mw, pop):
    return [
        build_preflop_s3_key_components(ip=ip,  oop=oop, stack_bb=stack,
                                        villain_profile=prof, exploit_setting=expl,
                                        multiway_context=mw, population_type=pop,
                                        action_context="VS_LIMP"),
        build_preflop_s3_key_components(ip=oop, oop=ip,  stack_bb=stack,
                                        villain_profile=prof, exploit_setting=expl,
                                        multiway_context=mw, population_type=pop,
                                        action_context="VS_LIMP"),
    ]

def deps_keys_for_vs_4bet(ip, oop, stack, prof, expl, mw, pop):
    return [
        build_preflop_s3_key_components(ip=ip,  oop=oop, stack_bb=stack,
                                        villain_profile=prof, exploit_setting=expl,
                                        multiway_context=mw, population_type=pop,
                                        action_context="VS_4BET"),
        build_preflop_s3_key_components(ip=oop, oop=ip,  stack_bb=stack,
                                        villain_profile=prof, exploit_setting=expl,
                                        multiway_context=mw, population_type=pop,
                                        action_context="VS_4BET"),
    ]

DEPS = {
    "OPEN":    deps_keys_for_open,
    "VS_3BET": deps_keys_for_vs_3bet,
    "VS_LIMP": deps_keys_for_vs_limp,
    "VS_4BET": deps_keys_for_vs_4bet,
}