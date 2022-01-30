import pytest
from src.evaluate import KptResults


def test_iou():
    n_misses_allowed = 10
    iou_threshold = 0.1
    kr = KptResults(n_misses_allowed, iou_threshold)
    """ Simple 2 by 2 boxes """
    # Empty intersection
    bbox_gt = (0, 0, 2, 2)
    bbox_p = (5, 5, 2, 2)
    assert(kr.get_accuracy_frame(bbox_gt, bbox_p) == 0.0)
    # Full intersection
    bbox_gt = (0, 0, 2, 2)
    bbox_p = (0, 0, 2, 2)
    assert(kr.get_accuracy_frame(bbox_gt, bbox_p) == 1.0)
    # Partial intersection
    bbox_gt = (0, 0, 2, 2)
    bbox_p = (1, 1, 2, 2)
    assert(kr.get_accuracy_frame(bbox_gt, bbox_p) == pytest.approx(0.142, 0.01))
    bbox_gt = (0, 0, 2, 2)
    bbox_p = (1, 0, 2, 2)
    assert(kr.get_accuracy_frame(bbox_gt, bbox_p) == pytest.approx(0.333, 0.01))
