"""Historical prototype retired after its event-label interpretation was corrected.

Response codes 2/4/8 encode rating magnitudes, not question identities. The old
prototype also used post-prompt samples and a within-person random split.
The original implementation remains in Git history for historical inspection.
"""
raise SystemExit(
    "This prototype is retired: response codes are rating magnitudes, not "
    "mind-wandering/concentration/tiredness categories. Use src/preprocess.py, "
    "src/extract_features.py and src/model_comparison.py for the validated "
    "event decoder and leave-one-subject-out evaluation."
)
