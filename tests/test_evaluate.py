import pytest
from src.evaluate import KptResults, EAO_Rank


def test_iou():
    n_misses_allowed = 10
    iou_threshold = 0.1
    kr = KptResults(n_misses_allowed, iou_threshold)
    # Empty intersection
    bbox_gt = (0, 0, 2, 2)
    bbox_p = (5, 5, 2, 2)
    assert(kr.get_iou(bbox_gt, bbox_p) == 0.0)
    bbox_gt = (0, 0, 2, 2)
    bbox_p = (2, 2, 2, 2)
    assert(kr.get_iou(bbox_gt, bbox_p) == 0.0)
    # Full intersection
    bbox_gt = (0, 0, 2, 2)
    bbox_p = (0, 0, 2, 2)
    assert(kr.get_iou(bbox_gt, bbox_p) == 1.0)
    bbox_gt = (5, 5, 10, 10)
    bbox_p = (5, 5, 10, 10)
    assert(kr.get_iou(bbox_gt, bbox_p) == 1.0)
    # Partial intersection
    bbox_gt = (0, 0, 2, 2)
    bbox_p = (1, 1, 2, 2)
    assert(kr.get_iou(bbox_gt, bbox_p) == pytest.approx(0.142, 0.01))
    bbox_gt = (0, 0, 2, 2)
    bbox_p = (1, 0, 2, 2)
    assert(kr.get_iou(bbox_gt, bbox_p) == pytest.approx(0.333, 0.01))


def test_robustness():
    n_misses_allowed = 10
    iou_threshold = 0.1
    kr = KptResults(n_misses_allowed, iou_threshold)
    kr.robustness_frames_counter = 25
    kr.n_visible = 40
    kr.n_excessive_frames = 10
    rob = kr.get_robustness_score()
    assert(rob == 0.5) # 25 / (40 + 10)


def test_accuracy():
    n_misses_allowed = 10
    iou_threshold = 0.1
    kr = KptResults(n_misses_allowed, iou_threshold)
    kr.iou_list = [1.0, 1.0, 0.5, 0.5]
    kr.n_visible = 10
    acc = kr.get_accuracy_score()
    assert(acc == 0.3) # 3. / 10


def test_EAO_Rank():
    # Empty sequence
    ss_list = []
    rank = EAO_Rank(0, 0)
    eao = rank.calculate_eao_score()
    assert(eao == 0.0)
    # Single sequence
    ss_list = [[1., 1., 1.]]
    rank = EAO_Rank(0, len(ss_list))
    rank.all_ss = ss_list
    rank.all_ss_len = [len(ss_list)]
    rank.all_ss_len_max = len(ss_list)
    eao = rank.calculate_eao_score()
    assert(eao == 1.0)
    ss_list = [[1., "ignore", 1.]]
    rank = EAO_Rank(0, len(ss_list))
    rank.all_ss = ss_list
    rank.all_ss_len = [len(ss_list)]
    rank.all_ss_len_max = len(ss_list)
    eao = rank.calculate_eao_score()
    assert(eao == 1.0)
    # Multiple sequences
    ss_list = [[1.],
               [0.]]
    rank = EAO_Rank(0, len(ss_list[0]))
    rank.all_ss = ss_list
    rank.all_ss_len = [1, 1]
    rank.all_ss_len_max = 1
    eao = rank.calculate_eao_score()
    assert(eao == 0.5)
    ss_list = [["ignore"],
               [0.7215]]
    rank = EAO_Rank(0, len(ss_list[0]))
    rank.all_ss = ss_list
    rank.all_ss_len = [1, 1]
    rank.all_ss_len_max = 1
    eao = rank.calculate_eao_score()
    assert(eao == 0.7215)
    # Test curve
    ss_list = [[0.3605, 0.5, "ignore", 0.],
               [0.0000, 0.0,       0., 0.7215]]
    rank = EAO_Rank(0, len(ss_list[1]))
    rank.all_ss = ss_list
    rank.all_ss_len = [4, 4]
    rank.all_ss_len_max = 4
    rank.calculate_eao_curve()
    assert(rank.eao_curve[0] == 0.18025)
    assert(rank.eao_curve[1] == 0.25)
    assert(rank.eao_curve[2] == 0.0)
    assert(rank.eao_curve[3] == 0.36075)
    eao = rank.calculate_eao_score()
    assert(eao == 0.19775)
