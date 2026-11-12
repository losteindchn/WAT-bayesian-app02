"""Retrospective first-Aha categories; query indices refer to displayed feedback."""
AHA_LOCATION_VERSION = "retrospective-first-aha-v1-20261111"


def aha_location_options(n_queries):
    if not isinstance(n_queries, int) or n_queries < 0:
        raise ValueError("n_queries must be a nonnegative integer")
    before = ("阅读谜题后、第一次查询之前" if n_queries else
              "阅读谜题后、进入答案页之前（本题未查询）")
    options = {"before_search": {"label": before, "category": "before_search", "query_index": None}}
    for i in range(1, n_queries + 1):
        options[f"after_feedback_{i}"] = {"label": f"看到第 {i} 次查询结果后",
                                           "category": "after_feedback", "query_index": i}
    for code, label in [("answer_phase", "结束查询后，在填写答案时"),
                        ("unknown", "有过这种感觉，但记不清具体在哪一步"),
                        ("none", "本题没有出现这种感觉")]:
        options[code] = {"label": label, "category": code, "query_index": None}
    return options


AHA_CONSISTENCY_VERSION = "aha-consistency-v1-20261111"
AHA_ABSENT_HIGH_THRESHOLD = 80


def aha_consistency_flags(location_category, aha_strength):
    """Review signal only; strength is the participant-facing 0-100 score."""
    if (location_category == "none" and isinstance(aha_strength, (int, float))
            and not isinstance(aha_strength, bool)
            and AHA_ABSENT_HIGH_THRESHOLD <= aha_strength <= 100):
        return ["no_aha_location_with_high_aha_strength"]
    return []
