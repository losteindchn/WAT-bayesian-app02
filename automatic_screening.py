"""Versioned operational timing rules, evaluated on completed formal trials."""
RULE_VERSION = "timing-v1-20261102"
DEFAULT_THRESHOLDS = {
    "stage1_timeout_trials": 3,
    "stage1_fast_consecutive": 5,
    "stage1_fast_trial_sec": 5,
    "stage2_fast_consecutive": 3,
    "stage2_fast_trial_sec": 10,
}


def automatic_invalidity(responses, thresholds=None):
    limits = dict(DEFAULT_THRESHOLDS)
    limits.update(thresholds or {})
    flags = []
    stage1 = [r for r in responses if r.get("stage") in ("stage1_passive_update", "stage1_timeout")]
    stage2 = [r for r in responses if r.get("stage") == "stage2_final_guess"]
    if sum(r.get("stage") == "stage1_timeout" for r in stage1) >= limits["stage1_timeout_trials"]:
        flags.append("automatic_stage1_repeated_timeout")
    if any(r.get("query_timeout_flag") and r.get("answer_timeout_flag") for r in stage2):
        flags.append("automatic_stage2_both_phases_timeout")
    for records, prefix, duration_key in [(stage1, "stage1", "stage1_trial_elapsed_sec"),
                                          (stage2, "stage2", "trial_elapsed_sec")]:
        streak = 0
        for row in records:
            duration = row.get(duration_key)
            fast = (isinstance(duration, (int, float)) and 0 <= duration < limits[prefix + "_fast_trial_sec"]
                    and not row.get("timeout_flag"))
            streak = streak + 1 if fast else 0
            if streak >= limits[prefix + "_fast_consecutive"]:
                flags.append("automatic_" + prefix + "_rapid_consecutive_trials")
                break
    return flags
