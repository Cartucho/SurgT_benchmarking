import pytest
from src.evaluate import KptResults


def test_iou():
    n_misses_allowed = 10
    iou_threshold = 0.1
    kr = KptResults(n_misses_allowed, iou_threshold)
    # Empty intersection
    bbox_gt = (0, 0, 1, 1)
    bbox_p = (5, 5, 1, 1)
    assert(kr.get_accuracy_frame(bbox_gt, bbox_p) == 0.0)
    # Full intersection
    bbox_gt = (0, 0, 1, 1)
    bbox_p = (0, 0, 1, 1)
    assert(kr.get_accuracy_frame(bbox_gt, bbox_p) == 1.0)
    # Partial intersection
    bbox_gt = (0, 0, 1, 1)
    bbox_p = (1, 1, 1, 1)
    assert(kr.get_accuracy_frame(bbox_gt, bbox_p) == pytest.approx(0.14, 0.1))
