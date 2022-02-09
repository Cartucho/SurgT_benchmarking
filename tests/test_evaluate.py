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
    rank = EAO_Rank()
    ss_list = []
    rank.append_ss_list(ss_list)
    eao = rank.calculate_eao_score()
    assert(eao == 0.0)
    # Single sequence
    rank = EAO_Rank()
    ss_list = [[1., 1., 1.]]
    rank.append_ss_list(ss_list)
    eao = rank.calculate_eao_score()
    assert(eao == 1.0)
    rank = EAO_Rank()
    ss_list = [[1., "is_difficult", 1.]]
    rank.append_ss_list(ss_list)
    eao = rank.calculate_eao_score()
    assert(eao == 1.0)
    rank = EAO_Rank()
    ## `is_difficult` at the end
    ss_list = [[1., 1., "is_difficult", "is_difficult"]]
    rank.append_ss_list(ss_list)
    eao = rank.calculate_eao_score()
    assert(eao == 1.0)
    # Multiple sequences
    rank = EAO_Rank()
    ss_list = [[1.],
               [0.]]
    rank.append_ss_list(ss_list)
    eao = rank.calculate_eao_score()
    assert(eao == 0.5)
    rank = EAO_Rank()
    ss_list = [[1., "is_difficult", "is_difficult"],
               [0.]]
    rank.append_ss_list(ss_list)
    eao = rank.calculate_eao_score()
    assert(eao == 0.5)
    rank = EAO_Rank()
    ss_list = [["is_difficult"],
               [0.7215]]
    rank.append_ss_list(ss_list)
    eao = rank.calculate_eao_score()
    assert(eao == 0.7215)
    # Test curve
    rank = EAO_Rank()
    ss_list = [[0.721, 0.5, 0., "is_difficult"],
               [0.000, 0.0, 0., 0.7215]]
    rank.append_ss_list(ss_list)
    rank.calculate_eao_curve()
    assert(rank.eao_curve[0] == 0.3605)
    assert(rank.eao_curve[1] == 0.25)
    assert(rank.eao_curve[2] == 0.0)
    assert(rank.eao_curve[3] == 0.7215)
    # Test append too
    rank = EAO_Rank()
    ss_list1 = [[1.0, "is_difficult"]]
    ss_list2 = [[1.0, 0.72]]
    ss_list3 = [[1.0, 0.72]]
    rank.append_ss_list(ss_list1)
    rank.append_ss_list(ss_list2)
    rank.append_ss_list(ss_list3)
    # Test ss_len_max
    rank = EAO_Rank()
    ss_list = [[1., 1., 1., "is_difficult", "is_difficult"],
               [0.]]
    rank.append_ss_list(ss_list)
    rank.update_ss_length()
    assert(rank.ss_len_max == 3) # "is_difficult" at the end does not count
