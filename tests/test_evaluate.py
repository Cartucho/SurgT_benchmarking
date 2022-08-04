import pytest
from src.evaluate import AnchorResults, EAO_Rank


def test_iou():
    n_misses_allowed = 10
    iou_threshold = 0.1
    err_3d_threshold = 1000
    ar = AnchorResults(n_misses_allowed, iou_threshold, err_3d_threshold)
    # Empty intersection
    bbox_gt = (0, 0, 2, 2)
    bbox_p = (5, 5, 2, 2)
    assert(ar.get_iou(bbox_gt, bbox_p) == 0.0)
    bbox_gt = (0, 0, 2, 2)
    bbox_p = (2, 2, 2, 2)
    assert(ar.get_iou(bbox_gt, bbox_p) == 0.0)
    # Full intersection
    bbox_gt = (0, 0, 2, 2)
    bbox_p = (0, 0, 2, 2)
    assert(ar.get_iou(bbox_gt, bbox_p) == 1.0)
    bbox_gt = (5, 5, 10, 10)
    bbox_p = (5, 5, 10, 10)
    assert(ar.get_iou(bbox_gt, bbox_p) == 1.0)
    # Partial intersection
    bbox_gt = (0, 0, 2, 2)
    bbox_p = (1, 1, 2, 2)
    assert(ar.get_iou(bbox_gt, bbox_p) == pytest.approx(0.142, 0.01))
    bbox_gt = (0, 0, 2, 2)
    bbox_p = (1, 0, 2, 2)
    assert(ar.get_iou(bbox_gt, bbox_p) == pytest.approx(0.333, 0.01))


def test_robustness():
    n_misses_allowed = 10
    iou_threshold = 0.1
    err_3d_threshold = 1000
    ar = AnchorResults(n_misses_allowed, iou_threshold, err_3d_threshold)
    ar.robustness_frames_counter = 25
    ar.n_visible_and_not_diff = 40
    ar.n_excessive_frames = 10
    rob = ar.get_robustness_score(25)
    assert(rob == 0.5) # 25 / (40 + 10)


def test_accuracy():
    n_misses_allowed = 10
    iou_threshold = 0.1
    err_3d_threshold = 1000
    ar = AnchorResults(n_misses_allowed, iou_threshold, err_3d_threshold)
    ar.iou_list = [1.0, 1.0, 0.5, 0.5]
    ar.n_visible_and_not_diff = 4
    acc = ar.get_accuracy_score()
    assert(acc == 0.75)


def test_EAO_Rank():
    # Empty sequence
    ss_list = []
    rank = EAO_Rank(0, 0)
    eao = rank.calculate_eao_score()
    assert(eao == 0.0)
    # Single sequence
    ss_list = [[1., 1., 1.]]
    rank = EAO_Rank(0, len(ss_list[0]))
    rank.final_ss = ss_list
    rank.all_ss_len_max = len(ss_list[0])
    eao = rank.calculate_eao_score()
    assert(eao == 1.0)
    ss_list = [[1., "ignore", 1.]]
    rank = EAO_Rank(0, len(ss_list[0]))
    rank.final_ss = ss_list
    rank.all_ss_len_max = len(ss_list[0])
    eao = rank.calculate_eao_score()
    assert(eao == 1.0)
    # Multiple sequences
    ss_list = [[1.],
               [0.]]
    rank = EAO_Rank(0, len(ss_list[0]))
    rank.final_ss = ss_list
    rank.all_ss_len_max = 1
    eao = rank.calculate_eao_score()
    assert(eao == 0.5)
    ss_list = [["ignore"],
               [0.7215]]
    rank = EAO_Rank(0, len(ss_list[0]))
    rank.final_ss = ss_list
    rank.all_ss_len_max = 1
    eao = rank.calculate_eao_score()
    assert(eao == 0.7215)
    ss_list = [[1.0],
               [0.75],
               [0.5],
               [0.25],
               [0.0]]
    rank = EAO_Rank(0, len(ss_list[0]))
    rank.final_ss = ss_list
    rank.all_ss_len_max = 1
    eao = rank.calculate_eao_score()
    assert(eao == 0.5)
    # Test curve
    ss_list = [[0.3605, 0.5, "ignore", 0.],
               [0.0000, 0.0,       0., 0.7215]]
    rank = EAO_Rank(0, len(ss_list[0]))
    eao_curve = rank.calculate_eao_curve(ss_list, len(ss_list[0]))
    assert(eao_curve[0] == 0.18025)
    assert(eao_curve[1] == 0.25)
    assert(eao_curve[2] == 0.0)
    assert(eao_curve[3] == 0.36075)
