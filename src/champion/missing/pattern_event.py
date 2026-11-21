"""Combine peer-calibrated missing interpolation and historical event curvature."""
from .event_bridge import predict_fold as apply_event
from .pattern_matched_pair import predict_fold as apply_peer


def predict_fold(train, released, targets, baseline, reference=None):
    with_peer = apply_peer(train, released, targets, baseline, reference=reference)
    return apply_event(train, released, targets, baseline, reference=with_peer)
