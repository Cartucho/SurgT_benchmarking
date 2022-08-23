import pytest
from src.evaluate import AnchorResults, EAO_Rank


""" Test Video class """
#TODO

""" Test Statistics class """
#TODO

""" Test EAO_Rank class """

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

""" Test SSeq class """
# TODO

""" Test KptSubSequences class """
# TODO

""" Test AnchorResults class """

def test_get_bbox_centr():
    ar = AnchorResults(0, 0, 0)
    bbox = (0, 0, 100, 100)
    centre = ar.get_bbox_centr(bbox)
    assert(centre[0] == 50 and centre[1] == 50)
    

def test_get_iou():
    ar = AnchorResults(0, 0, 0)
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


def test_get_robustness_score():
    ar = AnchorResults(0, 0, 0)
    # 25 / (40 + 10) = 25 / 50 = 0.5
    ar.n_visible_and_not_diff = 40
    ar.n_excessive_frames = 10
    rob = ar.get_robustness_score(25)
    assert(rob == 0.5)
    # 40 / (40 + 0) = 40 / 40 = 1.0
    ar.n_visible_and_not_diff = 40
    ar.n_excessive_frames = 0
    rob = ar.get_robustness_score(40)
    assert(rob == 1.0)


def test_get_accuracy_score():
    n_misses_allowed = 10
    iou_threshold = 0.1
    err_3d_threshold = 1000
    ar = AnchorResults(n_misses_allowed, iou_threshold, err_3d_threshold)
    ar.iou_list = [1.0, 1.0, 0.5, 0.5]
    acc = ar.get_accuracy_score()
    assert(acc == 0.75)
    ar.iou_list = [1.0, 1.0, "error_no_prediction", 0.5, 0.5]
    acc = ar.get_accuracy_score()
    assert(acc == 0.75)


def test_get_error_2D_score():
    ar = AnchorResults(0, 0, 0)
    ar.err_2d = [25.0, 5.0, 20.0, 50.0]
    err_2d_std, err_2d, n_f_2d = ar.get_error_2D_score()
    assert(err_2d_std == pytest.approx(16.2, 0.01))
    assert(err_2d == 25.)
    assert(n_f_2d == 4)
    ar.err_2d = [25.0, 5.0, "error_no_prediction", 20.0, 50.0]
    err_2d_std, err_2d, n_f_2d = ar.get_error_2D_score()
    assert(err_2d_std == pytest.approx(16.2, 0.01))
    assert(err_2d == 25.)
    assert(n_f_2d == 4)
    ar.err_2d = [30.0, 30.0, 30.0, 30.0, 30.0]
    err_2d_std, err_2d, n_f_2d = ar.get_error_2D_score()
    assert(err_2d_std == pytest.approx(0.0, 0.01))
    assert(err_2d == 30.)
    assert(n_f_2d == 5)


def test_get_error_3D_score():
    ar = AnchorResults(0, 0, 0)
    ar.err_3d = [5.0, 5.0, 10.0, 10.0]
    err_3d_std, err_3d, n_f_3d = ar.get_error_3D_score()
    assert(err_3d_std == 2.5)
    assert(err_3d == 7.5)
    assert(n_f_3d == 4)
    ar.err_3d = [5.0, 5.0, "error_no_prediction", 10.0, 10.0, "error_negative_disparity"]
    err_3d_std, err_3d, n_f_3d = ar.get_error_3D_score()
    assert(err_3d_std == 2.5)
    assert(err_3d == 7.5)
    assert(n_f_3d == 4)
    ar.err_3d = [2.0, 5.0, 5.0, 10.0, 10.0, 30.0]
    err_3d_std, err_3d, n_f_3d = ar.get_error_3D_score()
    assert(err_3d_std == pytest.approx(9.25, 0.01))
    assert(err_3d == pytest.approx(10.33, 0.01))
    assert(n_f_3d == 6)
    ar.err_3d = ["error_no_prediction", "error_no_prediction", "error_negative_disparity"]
    err_3d_std, err_3d, n_f_3d = ar.get_error_3D_score()
    assert(n_f_3d == 0)
    ar.err_3d = ["error_no_prediction", "error_no_prediction", 50.]
    err_3d_std, err_3d, n_f_3d = ar.get_error_3D_score()
    assert(err_3d_std == 0)
    assert(err_3d == 50.)
    assert(n_f_3d == 1)


""" Test class-less functions """
# TODO
