import pytest
import numpy as np
from src.evaluate import Statistics, AnchorResults, EAO_Rank


""" Test Video class """
#TODO

""" Test Statistics class """
def test_append_stats():
    stats = Statistics()
    # Creating stats to be appended
    stats_anchor = Statistics()
    ar = AnchorResults(10, 0.5, 1)
    ar.iou_list = [0.7, 0.1, 0, 0.8]
    ar.err_2d = [0, 51.25, 0, 2.1]
    ar.n_visible_and_not_diff = 10
    ar.n_excessive_frames = 10
    ar.rob_frames_counter_2d = 2
    ar.rob_frames_counter_3d = 3
    ar.err_3d = [2.7, 0.9, 1.1, 0.8, 1.0]
    ar.get_full_metric(stats_anchor)
    # Append stats and assert
    stats.append_stats(stats_anchor)
    assert stats.acc[0] == pytest.approx(0.4, 0.01)
    assert stats.n_f_rob[0] == 20 # n_visible_and_not_diff + n_excessive_frames
    assert stats.rob_2d[0] == pytest.approx(2./20, 0.01) # rob_frames_counter_2d / n_f_rob
    assert stats.rob_3d[0] == pytest.approx(3./20, 0.01) # rob_frames_counter_3d / n_f_rob
    assert stats.err_2d[0] == pytest.approx(13.34, 0.01)
    assert stats.err_2d_std[0] == pytest.approx(21.91, 0.01)
    assert stats.err_3d[0] == 1.3
    assert stats.err_3d_std[0] == pytest.approx(0.707, 0.01)
    assert stats.n_f_2d[0] == 4
    assert stats.n_f_3d[0] == 5


def test_merge_stats():
    stats = Statistics()
    for i in range(2): # Here we test the usage of two anchors with arbitrary values
        stats_anchor = Statistics()
        if i == 0:
            ar = AnchorResults(10, 0.5, 1)
            ar.iou_list = [0.7, 0.1, 0, 0.8] # Avg. 0.4
            ar.err_2d = [0, 51.25, 0, 2.1] # Avg. 13.34
            ar.err_3d = [2.7, 0.9, 1.1, 0.8, 1.0] # Avg. 1.3
            ar.n_visible_and_not_diff = 10
            ar.n_excessive_frames = 10
            ar.rob_frames_counter_2d = 2
            ar.rob_frames_counter_3d = 3
        elif i == 1:
            ar = AnchorResults(10, 0.6, 1)
            ar.iou_list = [0.5, 0.5, 0.6, 0.9, 0.8, 0.9] # Avg. 0.7
            ar.err_2d = [0.9, 12, 14, 2, 35, 25] # Avg. 14.82
            ar.err_3d = [2.7, 0.9, 1.1, 0.8, 1.0, 0.7] # Avg. 1.2
            ar.n_visible_and_not_diff = 7
            ar.n_excessive_frames = 3
            ar.rob_frames_counter_2d = 3
            ar.rob_frames_counter_3d = 4
        ar.get_full_metric(stats_anchor)
        stats.append_stats(stats_anchor)
    assert stats.n_f_rob == [20, 10] # n_visible_and_not_diff + n_excessive_frames: (10+10), (7+3)
    assert stats.rob_2d == [0.1, 0.3] # 2/20, 3/10
    assert stats.rob_3d == [0.15, 0.4] # 3/20, 4/10
    stats.merge_stats()
    assert stats.n_f_rob == 30
    assert stats.rob_2d == pytest.approx(1/6., 0.01) # 0.1 * (20/30) + 0.3 * (10/30)
    assert stats.rob_3d == pytest.approx(0.2333, 0.01) # 0.15 * (20/30) + 0.4 * (10/30)
    assert stats.n_f_2d == 10 # 4 + 6
    assert stats.acc == pytest.approx(0.58, 0.01) # 0.4 * (4/10) + 0.7 * (6/10)
    assert stats.err_2d == pytest.approx(14.228, 0.01) # 13.34 * (4/10) + 14.82 * (6/10)
    assert stats.err_2d_std == pytest.approx(16.019, 0.01)
    assert stats.n_f_3d == 11 # 5 + 6
    assert stats.err_3d == pytest.approx(1.245, 0.01) # 1.3 * (5/11) + 1.2 * (6/11)
    assert stats.err_3d_std == pytest.approx(0.694, 0.01)


""" Test EAO_Rank class """

def test_EAO_Rank():
    # Empty sequence
    ss_list = []
    rank = EAO_Rank(0, 0)
    eao = rank.calculate_eao_score()
    assert eao == 0.0
    # Single sequence
    ss_list = [[1., 1., 1.]]
    rank = EAO_Rank(0, len(ss_list[0]))
    rank.final_ss = ss_list
    rank.all_ss_len_max = len(ss_list[0])
    eao = rank.calculate_eao_score()
    assert eao == 1.0
    ss_list = [[1., "ignore", 1.]]
    rank = EAO_Rank(0, len(ss_list[0]))
    rank.final_ss = ss_list
    rank.all_ss_len_max = len(ss_list[0])
    eao = rank.calculate_eao_score()
    assert eao == 1.0
    # Multiple sequences
    ss_list = [[1.],
               [0.]]
    rank = EAO_Rank(0, len(ss_list[0]))
    rank.final_ss = ss_list
    rank.all_ss_len_max = 1
    eao = rank.calculate_eao_score()
    assert eao == 0.5
    ss_list = [["ignore"],
               [0.7215]]
    rank = EAO_Rank(0, len(ss_list[0]))
    rank.final_ss = ss_list
    rank.all_ss_len_max = 1
    eao = rank.calculate_eao_score()
    assert eao == 0.7215
    ss_list = [[1.0],
               [0.75],
               [0.5],
               [0.25],
               [0.0]]
    rank = EAO_Rank(0, len(ss_list[0]))
    rank.final_ss = ss_list
    rank.all_ss_len_max = 1
    eao = rank.calculate_eao_score()
    assert eao == 0.5
    # Test curve
    ss_list = [[0.3605, 0.5, "ignore", 0.],
               [0.0000, 0.0,       0., 0.7215]]
    rank = EAO_Rank(0, len(ss_list[0]))
    eao_curve = rank.calculate_eao_curve(ss_list, len(ss_list[0]))
    assert eao_curve[0] == 0.18025
    assert eao_curve[1] == 0.25
    assert eao_curve[2] == 0.0
    assert eao_curve[3] == 0.36075

""" Test SSeq class """
# TODO

""" Test KptSubSequences class """
# TODO

""" Test AnchorResults class """
def test_calculate_bbox_metrics():
    ar = AnchorResults(10, 0.1, 100)
    bbox1_gt = (50, 50, 50, 50)
    bbox1_p = None
    bbox2_gt = (50, 50, 50, 50)
    bbox2_p = (50, 50, 50, 50)
    flag_track_fail_2d, flag_track_fail_3d, iou = ar.calculate_bbox_metrics(bbox1_gt, bbox1_p, bbox2_gt, bbox2_p, False, False)
    assert flag_track_fail_2d == False
    assert flag_track_fail_3d == False
    assert iou == 0
    assert ar.n_misses_successive_2d == 1
    assert ar.n_misses_successive_3d == 1
    assert ar.iou_list[0] == "error_no_prediction"
    assert ar.err_2d[0] == "error_no_prediction"
    assert ar.err_3d[0] == "error_no_prediction"
    bbox1_p = (50, 50, 50, 50) # Replace None, with the correct bbox
    ar.calculate_bbox_metrics(bbox1_gt, bbox1_p, bbox2_gt, bbox2_p, False, False)
    assert ar.iou_list[1] == 1.
    assert ar.n_misses_successive_2d == 0
    assert ar.n_misses_successive_3d == 2 # Since disparity was still 0


def test_use_scores_before_failure_2d():
    ar = AnchorResults(10, 0, 0) # setting 10 to n_misses_allowed
    ar.iou_list = [0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    l = 10000
    ar.err_2d = [5., l, l, l, l, l, l, l, l, l, l] # large error 10 times, trigger 3D fail
    ar.use_scores_before_failure_2d()
    assert ar.iou_list == [0.5]
    assert ar.err_2d == [5]


def test_use_scores_before_failure_3d():
    ar = AnchorResults(10, 0, 0) # setting 10 to n_misses_allowed
    l = 10000
    ar.err_3d = [5., l, l, l, l, l, l, l, l, l, l] # large error 10 times, trigger 3D fail
    ar.use_scores_before_failure_3d()
    assert ar.err_3d == [5]


def test_get_bbox_centr():
    ar = AnchorResults(0, 0, 0)
    bbox = (0, 0, 100, 100)
    centre = ar.get_bbox_centr(bbox)
    assert centre[0] == 50
    assert centre[1] == 50
    bbox = (50, 10, 50, 100)
    centre = ar.get_bbox_centr(bbox)
    assert centre[0] == 75.
    assert centre[1] == 60.
    bbox = (50, 50, 1, 1)
    centre = ar.get_bbox_centr(bbox)
    assert centre[0] == 50.5
    assert centre[1] == 50.5


def test_get_l2_norm():
    ar = AnchorResults(0, 0, 0)
    err_2d = ar.get_l2_norm(np.array([10., 50.]), np.array([20., 50.]))
    assert err_2d == pytest.approx(10., 0.01)
    err_3d = ar.get_l2_norm(np.array([10., 10., 10.]), np.array([11., 12., 13.]))
    assert err_3d == pytest.approx(3.74, 0.01)


def test_get_3d_pt():
    ar = AnchorResults(0, 0, 0)
    ar.Q = np.array([[1., 0., 0., -400.],
                     [0., 1., 0., -500.],
                     [0., 0., 0.,  300.],
                     [0., 0., 0.2,   0.]])
    pt_3d = ar.get_3d_pt(40, 10, 10)
    assert pt_3d[0] == pytest.approx(-48.75, 0.01)
    assert pt_3d[1] == pytest.approx(-61.2, 0.01)
    z = (ar.Q[2, 3] * (1./ar.Q[3, 2])) / 40. # f*b/disp
    assert pt_3d[2] == pytest.approx(z, 0.01)


def test_l2_norm_errors():
    ar = AnchorResults(10, 0.1, 100)
    bbox1_gt = (50, 50, 50, 50)
    bbox1_p = (50, 50, 50, 50)
    bbox2_gt = (50, 50, 50, 50)
    bbox2_p = (50, 50, 50, 50)
    ar.calculate_l2_norm_errors(bbox1_gt, bbox1_p, bbox2_gt, bbox2_p, True, True)
    assert not ar.err_2d
    assert not ar.err_3d
    ar.calculate_l2_norm_errors(bbox1_gt, bbox1_p, bbox2_gt, bbox2_p, False, False)
    assert ar.err_2d[0] == 0
    assert ar.n_misses_successive_2d == 0
    assert ar.err_3d[0] == "error_non_positive_disp"
    assert ar.n_misses_successive_3d == 1


def test_get_iou():
    ar = AnchorResults(0, 0, 0)
    # Empty intersection
    bbox_gt = (0, 0, 2, 2)
    bbox_p = (5, 5, 2, 2)
    assert ar.get_iou(bbox_gt, bbox_p) == 0.0
    bbox_gt = (0, 0, 2, 2)
    bbox_p = (2, 2, 2, 2)
    assert ar.get_iou(bbox_gt, bbox_p) == 0.0
    # Full intersection
    bbox_gt = (0, 0, 2, 2)
    bbox_p = (0, 0, 2, 2)
    assert ar.get_iou(bbox_gt, bbox_p) == 1.0
    bbox_gt = (5, 5, 10, 10)
    bbox_p = (5, 5, 10, 10)
    assert ar.get_iou(bbox_gt, bbox_p) == 1.0
    # Partial intersection
    bbox_gt = (0, 0, 2, 2)
    bbox_p = (1, 1, 2, 2)
    assert ar.get_iou(bbox_gt, bbox_p) == pytest.approx(0.142, 0.01)
    bbox_gt = (0, 0, 2, 2)
    bbox_p = (1, 0, 2, 2)
    assert ar.get_iou(bbox_gt, bbox_p) == pytest.approx(0.333, 0.01)


def test_get_robustness_score():
    ar = AnchorResults(0, 0, 0)
    # 25 / (40 + 10) = 25 / 50 = 0.5
    ar.n_visible_and_not_diff = 40
    ar.n_excessive_frames = 10
    rob = ar.get_robustness_score(25)
    assert rob == 0.5
    # 40 / (40 + 0) = 40 / 40 = 1.0
    ar.n_visible_and_not_diff = 40
    ar.n_excessive_frames = 0
    rob = ar.get_robustness_score(40)
    assert rob == 1.0


def test_get_accuracy_score():
    n_misses_allowed = 10
    iou_threshold = 0.1
    err_3d_threshold = 1000
    ar = AnchorResults(n_misses_allowed, iou_threshold, err_3d_threshold)
    ar.iou_list = [1.0, 1.0, 0.5, 0.5]
    acc = ar.get_accuracy_score()
    assert acc == 0.75
    ar.iou_list = [1.0, 1.0, "error_no_prediction", 0.5, 0.5]
    acc = ar.get_accuracy_score()
    assert acc == 0.75


def test_get_error_2D_score():
    ar = AnchorResults(0, 0, 0)
    ar.err_2d = [25.0, 5.0, 20.0, 50.0]
    err_2d_std, err_2d, n_f_2d = ar.get_error_2D_score()
    assert err_2d_std == pytest.approx(16.2, 0.01)
    assert err_2d == 25.
    assert n_f_2d == 4
    ar.err_2d = [25.0, 5.0, "error_no_prediction", 20.0, 50.0]
    err_2d_std, err_2d, n_f_2d = ar.get_error_2D_score()
    assert err_2d_std == pytest.approx(16.2, 0.01)
    assert err_2d == 25.
    assert n_f_2d == 4
    ar.err_2d = [30.0, 30.0, 30.0, 30.0, 30.0]
    err_2d_std, err_2d, n_f_2d = ar.get_error_2D_score()
    assert err_2d_std == pytest.approx(0.0, 0.01)
    assert err_2d == 30.
    assert n_f_2d == 5


def test_get_error_3D_score():
    ar = AnchorResults(0, 0, 0)
    ar.err_3d = [5.0, 5.0, 10.0, 10.0]
    err_3d_std, err_3d, n_f_3d = ar.get_error_3D_score()
    assert err_3d_std == 2.5
    assert err_3d == 7.5
    assert n_f_3d == 4
    ar.err_3d = [5.0, 5.0, "error_no_prediction", 10.0, 10.0, "error_non_positive_disp"]
    err_3d_std, err_3d, n_f_3d = ar.get_error_3D_score()
    assert err_3d_std == 2.5
    assert err_3d == 7.5
    assert n_f_3d == 4
    ar.err_3d = [2.0, 5.0, 5.0, 10.0, 10.0, 30.0]
    err_3d_std, err_3d, n_f_3d = ar.get_error_3D_score()
    assert err_3d_std == pytest.approx(9.25, 0.01)
    assert err_3d == pytest.approx(10.33, 0.01)
    assert n_f_3d == 6
    ar.err_3d = ["error_no_prediction", "error_no_prediction", "error_non_positive_disp"]
    err_3d_std, err_3d, n_f_3d = ar.get_error_3D_score()
    assert n_f_3d == 0
    ar.err_3d = ["error_no_prediction", "error_no_prediction", 50.]
    err_3d_std, err_3d, n_f_3d = ar.get_error_3D_score()
    assert err_3d_std == 0
    assert err_3d == 50.
    assert n_f_3d == 1


""" Test class-less functions """
# TODO
