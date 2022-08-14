import os
from src import utils
from src.sample_tracker import Tracker

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class Video:
    def __init__(self, case_sample_path, is_to_rectify):
        # Load video info
        self.case_sample_path = case_sample_path
        video_info_path = os.path.join(case_sample_path, "info.yaml")
        video_info = utils.load_yaml_data(video_info_path)
        #print(video_info)
        self.stack_type = video_info["video_stack"]
        self.im_height = video_info["resolution"]["height"]
        self.im_width = video_info["resolution"]["width"]
        # Load rectification data
        self.is_to_rectify = is_to_rectify
        self.calib_path = os.path.join(case_sample_path, "calibration.yaml")
        utils.is_path_file(self.calib_path)
        self.load_calib_data()
        self.stereo_rectify()
        if is_to_rectify:
            self.get_rectification_maps()
        # Load video
        name_video = video_info["name_video"]
        self.video_path = os.path.join(case_sample_path, name_video)
        #print(self.video_path)
        self.video_restart()
        # Load ground-truth
        self.gt_files = video_info["name_ground_truth"]
        self.n_keypoints = len(self.gt_files)


    def video_restart(self):
        self.cap = cv.VideoCapture(self.video_path)
        self.frame_counter = -1 # So that the first get_frame() goes to zero


    def load_ground_truth(self, ind_kpt):
        gt_data_path = os.path.join(self.case_sample_path, self.gt_files[ind_kpt])
        self.gt_data = utils.load_yaml_data(gt_data_path)


    def get_bbox_gt(self, frame_counter):
        """
            Return two bboxes in format (u, v, width, height)

                                 (u,)   (u + width,)
                          (0,0)---.--------.---->
                            |
                       (,v) -     x--------.
                            |     |  bbox  |
              (,v + height) -     .________.
                            v

            Note: we assume that the gt coordinates are already set for the
                  rectified images, otherwise we would have to re-map these coordinates.
        """
        bbox_1 = None
        bbox_2 = None
        is_visible_in_both_stereo, is_difficult, bbxs = self.gt_data[frame_counter]
        if bbxs is not None:
            bbox_1 = bbxs[0]
            bbox_2 = bbxs[1]
        return bbox_1, bbox_2, is_difficult, is_visible_in_both_stereo


    def is_bbox_inside_image(self, bbox_1, bbox_2):
        u, u_max, v, v_max = bbox_1[:]
        if u < 0 or v < 0 or u >= self.im_width or v >= self.im_height:
            return False
        u, u_max, v, v_max = bbox_2[:]
        if u < 0 or v < 0 or u >= self.im_width or v >= self.im_height:
            return False
        return True


    def load_calib_data(self):
        fs = cv.FileStorage(self.calib_path, cv.FILE_STORAGE_READ)
        self.r = np.array(fs.getNode('R').mat(), dtype=np.float64)
        self.t = np.array(fs.getNode('T').mat()[0], dtype=np.float64)
        self.m1 = np.array(fs.getNode('M1').mat(), dtype=np.float64)
        self.d1 = np.array(fs.getNode('D1').mat()[0], dtype=np.float64)
        self.m2 = np.array(fs.getNode('M2').mat(), dtype=np.float64)
        self.d2 = np.array(fs.getNode('D2').mat()[0], dtype=np.float64)


    def stereo_rectify(self):
        self.R1, self.R2, self.P1, self.P2, self.Q, _roi1, _roi2 = \
            cv.stereoRectify(cameraMatrix1=self.m1,
                             distCoeffs1=self.d1,
                             cameraMatrix2=self.m2,
                             distCoeffs2=self.d2,
                             imageSize=(self.im_width, self.im_height),
                             R=self.r,
                             T=self.t,
                             flags=cv.CALIB_ZERO_DISPARITY,
                             alpha=0.0
                            )


    def get_rectification_maps(self):
        self.map1_x, self.map1_y = \
            cv.initUndistortRectifyMap(cameraMatrix=self.m1,
                                       distCoeffs=self.d1,
                                       R=self.R1,
                                       newCameraMatrix=self.P1,
                                       size=(self.im_width, self.im_height),
                                       m1type=cv.CV_32FC1
                                      )

        self.map2_x, self.map2_y = \
            cv.initUndistortRectifyMap(
                                       cameraMatrix=self.m2,
                                       distCoeffs=self.d2,
                                       R=self.R2,
                                       newCameraMatrix=self.P2,
                                       size=(self.im_width, self.im_height),
                                       m1type=cv.CV_32FC1
                                      )


    def split_frame(self, frame):
        if self.stack_type == "vertical":
            im1 = frame[:self.im_height, :]
            im2 = frame[self.im_height:, :]
        elif self.stack_type == "horizontal":
            im1 = frame[:, :self.im_width]
            im2 = frame[:, self.im_width:]
        else:
            print("Error: unrecognized stack type `{}`!".format(stack_type))
            exit()
        if self.is_to_rectify:
            im1 = cv.remap(im1, self.map1_x, self.map1_y, cv.INTER_LINEAR)
            im2 = cv.remap(im2, self.map2_x, self.map2_y, cv.INTER_LINEAR)
        return im1, im2


    def get_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        self.cap.release()
        return None


    def stop_video(self):
        self.cap.release()


    def get_terminator_frame(self):
        # Go through entire video and find the last frame whose,
        # bbox is (1) visible and (2) not difficult
        terminator_frame = 0
        if self.gt_data is not None:
            last_frame = len(self.gt_data) - 1 # -1 since we start from 0
            # Since we are looking for the last good frame, we go from last_frame until 0
            for i in range(last_frame, -1, -1): # [last_frame, last_frame - 1, ..., 1, 0]
                is_visible_in_both_stereo, is_difficult, bbxs = self.gt_data[i]
                if is_visible_in_both_stereo and \
                   not is_difficult:
                   terminator_frame = i
                   break
        return terminator_frame


class Statistics:
    def __init__(self):
        self.acc = []
        self.rob_2d = []
        self.err_2d = []
        self.rob_3d = []
        self.err_3d = []
        self.n_f_2d = [] # n_f -> number of frames
        self.n_f_rob = []
        self.n_f_3d = []


    def append_stats(self, stats):
        self.acc.append(stats.acc)
        self.rob_2d.append(stats.rob_2d)
        self.err_2d.append(stats.err_2d)
        self.rob_3d.append(stats.rob_3d)
        self.err_3d.append(stats.err_3d)
        self.n_f_2d.append(stats.n_f_2d)
        self.n_f_rob.append(stats.n_f_rob)
        self.n_f_3d.append(stats.n_f_3d)


    def merge_stats(self):
        """ Merge scores using weighted average """
        self.acc = np.ma.average(self.acc, weights=self.n_f_2d)
        self.rob_2d = np.ma.average(self.rob_2d, weights=self.n_f_rob)
        self.err_2d = np.ma.average(self.err_2d, weights=self.n_f_2d)
        self.rob_3d = np.ma.average(self.rob_3d, weights=self.n_f_rob)
        self.err_3d = np.ma.average(self.err_3d, weights=self.n_f_3d)
        """ Merge scores by summing all the frames """
        self.n_f_2d = sum(self.n_f_2d)
        self.n_f_rob = sum(self.n_f_rob)
        self.n_f_3d = sum(self.n_f_3d)


class EAO_Rank:
    def __init__(self, N_min, N_max):
        self.final_ss = []
        self.all_ss_len = []
        self.all_ss_len_max = 0
        self.N_min = N_min
        self.N_max = N_max


    def add_kpt_ss(self, kss):
        """
            Append a single mean IoU score, instead of all the sub-sequences of a keypoint.
             The reason is that we want to avoid having a longer videos (with many sub-sequences),
             having a too big impact in the final EAO score. Therefore, we can first merge all the
             sub-sequences. This way, we garantee that each keypoint per video, contributes with a single sub-sequence.
        """
        all_kps_ss = []
        all_kps_ss_len_max = 0
        for ss in kss.kpt_all_ss:
            all_kps_ss.append(ss.ss_iou_scores)
            ss_len = len(ss.ss_iou_scores)
            self.all_ss_len.append(ss_len) # Still, we want to keep track of all the individual ss_len, for calculating N_min and N_max
            all_kps_ss_len_max = max(all_kps_ss_len_max, ss_len)
        self.all_ss_len_max = max(self.all_ss_len_max, all_kps_ss_len_max)
        # Calculate the mean IoU scores, to create a single ss to used in the final EAO score
        mean_kpt_iou_scores = self.calculate_eao_curve(all_kps_ss, all_kps_ss_len_max)
        #print("Case sample: {}, Kpt_id: {}, Mean IoU: {}".format(kss.case_sample_path, kss.kpt_id, mean_kpt_iou_scores))
        self.final_ss.append(mean_kpt_iou_scores)


    def calculate_N_min_and_N_max(self):
        ss_len_mean = np.mean(self.all_ss_len)
        ss_len_std = np.std(self.all_ss_len)
        N_min = max(1, int(round(ss_len_mean - ss_len_std)))
        N_max = int(round(ss_len_mean + ss_len_std))
        print("Mean:{} Std:{} N_min:{} N_max:{}".format(ss_len_mean,
                                                        ss_len_std,
                                                        N_min,
                                                        N_max))
        # Show histogram
        bins = int(self.all_ss_len_max / 10) # Make bars of 10 frames
        hist, bin_edges = np.histogram(self.all_ss_len, bins=bins)
        _ = plt.hist(self.all_ss_len, bins=bin_edges)  # arguments are passed to np.histogram
        plt.show()


    def calculate_eao_curve(self, all_ss, all_ss_len_max):
        eao_curve = []
        for i in range(all_ss_len_max):
            ss_sum = 0.0
            ss_counter = 0
            for ss in all_ss:
                if len(ss) > i:
                    if ss[i] == "ignore":
                        continue
                    ss_sum += ss[i]
                    ss_counter += 1
            if ss_counter == 0:
                # This happens when all of the ss had the value "ignore" for frame i
                # , which happens when the bbox is not visible, or difficult
                eao_curve.append("ignore")
                continue
            score = ss_sum / ss_counter
            eao_curve.append(score)
        return eao_curve


    def calculate_eao_score(self):
        eao_curve = self.calculate_eao_curve(self.final_ss, self.all_ss_len_max)
        if not eao_curve:
            # If empty list
            return 0.0
        eao_curve_N = eao_curve[self.N_min:self.N_max]
        # Remove any "is_difficult" score
        eao_curve_N_filt = [value for value in eao_curve_N if value != "ignore"]
        return np.mean(eao_curve_N_filt)


class SSeq:
    """
        A specific sub-sequence
    """
    def __init__(self):
        self.ss_iou_scores = []


    def add_iou_score(self, iou):
        self.ss_iou_scores.append(iou)


class KptSubSequences:
    """
       All the sub-sequences of a specific keypoint on a specific video.
         Example with a video of 1000 frames.

       (a) A sub-sequence is created when the tracker is initialized.
           The tracker is initialized at each anchor (the video is repeated for each anchor).
           And all the sub-sequences go until the TERMINATOR_FRAME (explained in (b)).

              (1) tracker initialized,
                  at the first anchor
                       |
                       |      (2) tracker initialized,
                       |          at second anchor
                       |            |
        video   0      v            v                       999
        frames: |------x------------x------------------------>
                       |            |
                       |-------------------------------------> sub-sequence 1
                                    |
                                    |----------------------- > sub-sequence 2

       (b) All the sub-sequence stop at the TERMINATOR_FRAME.

           The TERMINATOR_FRAME, is the last frame of the video, for a specific keypoint,
             where we have a bbox that is both (i) visible, and (ii) not marked as difficult.

              (1) first anchor
                       |
                       |    (2) second
                       |        anchor    (3) TERMINATOR_FRAME
                       |           |                |
        video   0      v           v                v       999
        frames: |------x-----------x----------------x------->
                       |           |                |
                       |----------------------------> sub-sequence 1
                                   |                |
                                   |----------------> sub-sequence 2


       (c) During the video a sub-sequence stores the IoU scores at each frame.
           If in a frame the bbox is not visible or is difficult it stores, `ignore` instead,
             so that this frame is ignored when calculating the final EAO score.
           When the bbox is marker as `is_difficult` it is also ignored.

              (1) first anchor
                       |
                       |      (2) bbox becomes
                       |         not visible
                       |            |        (3) bbox
                       |            |            visible again
                       |            |             |
                0      v            v             v         999
        frames: |------x------------x-------------x---------->
                       |            |             |
                       |------------x-------------x----------> sub-sequence 1
        IoU scores:    [1, 0.9...0.5, i, i, i... i,  0.4  0.3]
                                    | i = `ignore`|

        (d) If a tracker fails the rest the previous sub-sequence is padded with zeros.

              (1) first anchor
                       |
                       |            (2) tracker
                       |                failure
                       |                  |
        video   0      v                  v                 999
        frames: |------x------------------x------------------>
                       |                  |
                       |------------------x------------------> sub-sequence 1
        IoU scores 1:  [1,0.8...0.003,0.002, 0, 0,..., 0, 0, 0]
                                          |   zero padding   |


        The sub-sequences start with high scores, since
          sub-sequences are created when the tracker is initialized
    """
    def __init__(self, terminator_frame, case_sample_path, kpt_id):
        self.TERMINATOR_FRAME = terminator_frame
        self.case_sample_path = case_sample_path
        self.kpt_id = kpt_id
        self.kpt_all_ss = []


    def add_ss(self):
        ss = SSeq()
        self.kpt_all_ss.append(ss)


    def add_iou_score(self, iou, frame_counter):
        if frame_counter <= self.TERMINATOR_FRAME:
            ss_last = self.kpt_all_ss[-1]
            ss_last.add_iou_score(iou)



class AnchorResults:
    def __init__(self, n_misses_allowed, iou_threshold, err_3d_threshold, Q=None):
        self.n_misses_allowed = n_misses_allowed
        self.iou_threshold = iou_threshold
        self.err_3d_threshold = err_3d_threshold
        self.Q = Q
        self.iou_list = []
        self.err_2d = []
        self.err_3d = []
        self.rob_frames_counter_2d = 0
        self.rob_frames_counter_3d = 0
        self.n_excessive_frames = 0
        self.n_visible_and_not_diff = 0
        self.n_misses_successive_2d = 0
        self.n_misses_successive_3d = 0


    def calculate_bbox_metrics(self, bbox1_gt, bbox1_p, bbox2_gt, bbox2_p, is_track_fail_2d, is_track_fail_3d):
        """
            Compute scores for a specific image frame of a video, given the ground truth and predictions.
        """
        iou = 0
        if bbox1_p is not None and bbox2_p is not None:
            # Tracker predicted the position of the bounding boxes
            self.calculate_l2_norm_errors(bbox1_gt, bbox1_p, bbox2_gt, bbox2_p, is_track_fail_2d, is_track_fail_3d)
            if not is_track_fail_2d:
                iou1 = self.get_iou(bbox1_gt, bbox1_p)
                iou2 = self.get_iou(bbox2_gt, bbox2_p)
                iou = np.mean([iou1, iou2])
                self.iou_list.append(iou)
                if iou1 > self.iou_threshold and iou2 > self.iou_threshold:
                    # Enough overlap
                    self.rob_frames_counter_2d += 1
                    self.n_misses_successive_2d = 0
                else:
                    # Not enough overlap
                    self.n_misses_successive_2d += 1
        else:
            self.iou_list.append("error_no_prediction")
            self.err_2d.append("error_no_prediction")
            self.err_3d.append("error_no_prediction")
            # Tracker failed to predict the bounding boxes
            self.n_misses_successive_2d += 1
            self.n_misses_successive_3d += 1
        # If track didn't fail yet, check if it did in this current image frame
        flag_track_fail_2d = False
        if not is_track_fail_2d:
            flag_track_fail_2d = (self.n_misses_successive_2d > self.n_misses_allowed)
        flag_track_fail_3d = False
        if not is_track_fail_3d:
            flag_track_fail_3d = (self.n_misses_successive_3d > self.n_misses_allowed)
        return flag_track_fail_2d, flag_track_fail_3d, iou


    def use_scores_before_failure_2d(self):
        """
            Delete the last `n_misses_allowed + 1` scores, since tracker failed,
              so that the tracking failure does not affect the scores.
        """
        n_scores_to_delete = self.n_misses_allowed + 1
        self.iou_list = self.iou_list[:-n_scores_to_delete]
        self.err_2d = self.err_2d[:-n_scores_to_delete]


    def use_scores_before_failure_3d(self):
        n_scores_to_delete = self.n_misses_allowed + 1
        self.err_3d = self.err_3d[:-n_scores_to_delete]


    def get_bbox_centr(self, bbox):
        centr_u = int(bbox[0] + (bbox[2] / 2))
        centr_v = int(bbox[1] + (bbox[3] / 2))
        return np.array([centr_u, centr_v])


    def get_l2_norm(self, centr_gt, centr_p):
        return np.linalg.norm(centr_gt - centr_p)


    def get_3d_pt(self, disp, u, v):
        assert(disp > 0)
        pt_2d = np.array([[u],
                          [v],
                          [disp],
                          [1.0]
                          ], dtype=np.float32)
        pt_3d = np.matmul(self.Q, pt_2d)
        pt_3d /= pt_3d[3, 0]
        return pt_3d


    def calculate_l2_norm_errors(self, bbox1_gt, bbox1_p, bbox2_gt, bbox2_p, is_track_fail_2d, is_track_fail_3d):
        centr_2d_gt_1 = self.get_bbox_centr(bbox1_gt)
        centr_2d_p_1 = self.get_bbox_centr(bbox1_p)
        centr_2d_gt_2 = self.get_bbox_centr(bbox2_gt)
        centr_2d_p_2 = self.get_bbox_centr(bbox2_p)
        # Get 2D error [pixels]
        if not is_track_fail_2d:
            err_2d_1 = self.get_l2_norm(centr_2d_gt_1, centr_2d_p_1)
            err_2d_2 = self.get_l2_norm(centr_2d_gt_2, centr_2d_p_2)
            err_2d = np.mean([err_2d_1, err_2d_2])
            self.err_2d.append(err_2d)
        # Get 3D error [mm]
        if not is_track_fail_3d:
            disp_p = centr_2d_p_1[0] - centr_2d_p_2[0]
            disp_gt = centr_2d_gt_1[0] - centr_2d_gt_2[0]
            if disp_p > 0 and disp_gt > 0:
                """
                 I am assuming that `centr_bbox1_p` and `centr_bbox2_p` have the same `v`,
                 which should be the case for a stereo Tracker that works with rectified images as input.
                """
                centr_3d_p = self.get_3d_pt(disp_p, centr_2d_p_1[0], centr_2d_p_1[1])
                centr_3d_gt = self.get_3d_pt(disp_gt, centr_2d_gt_1[0], centr_2d_gt_1[1])
                err_3d = self.get_l2_norm(centr_3d_p, centr_3d_gt)
                self.err_3d.append(err_3d)
                if err_3d <= self.err_3d_threshold:
                    self.rob_frames_counter_3d += 1
                    self.n_misses_successive_3d = 0
                else:
                    self.n_misses_successive_3d += 1
            else:
                self.err_3d.append("error_negative_disparity")
                self.n_misses_successive_3d += 1


    def get_iou(self, bbox_gt, bbox_p):
        gt_left, gt_top, gt_right, gt_bot = [bbox_gt[0], bbox_gt[1], bbox_gt[0]+bbox_gt[2], bbox_gt[1]+bbox_gt[3]]
        p_left, p_top, p_right, p_bot = [bbox_p[0], bbox_p[1], bbox_p[0]+bbox_p[2], bbox_p[1]+bbox_p[3]]
        inter_left = max(gt_left, p_left)
        inter_top = max(gt_top, p_top)
        inter_right = min(gt_right, p_right)
        inter_bot = min(gt_bot, p_bot)
        inter_width = np.maximum(0,inter_right - inter_left)
        inter_height = np.maximum(0,inter_bot - inter_top)
        inter_area = inter_width * inter_height
        gt_width = bbox_gt[2]
        gt_height = bbox_gt[3]
        p_width = bbox_p[2]
        p_height = bbox_p[3]
        gt_area = gt_width * gt_height
        p_area = p_width * p_height
        union_area = gt_area + p_area - inter_area
        iou = inter_area / float(union_area)
        assert(iou >= 0.0 and iou <= 1.0)
        return iou


    def get_accuracy_score(self):
        acc = 1.0
        iou_list_filtered = [value for value in self.iou_list if value != "error_no_prediction"]
        if self.n_visible_and_not_diff > 0:
            acc = np.mean(iou_list_filtered)
        assert(acc >= 0.0 and acc <= 1.0)
        return acc


    def get_robustness_score(self, rob_frames_counter):
        rob = 1.0
        denominator = self.n_visible_and_not_diff + self.n_excessive_frames
        if denominator > 0:
            rob = rob_frames_counter / denominator
        assert(rob >= 0.0 and rob <= 1.0)
        return rob


    def get_full_metric(self, stats_anchor):
        """
        Only happens after all frames are processed, end of video for-loop!
        """
        assert(len(self.iou_list) == len(self.err_2d))
        acc = self.get_accuracy_score()
        rob_2d = self.get_robustness_score(self.rob_frames_counter_2d)
        err_filtered_2d = [value for value in self.err_2d if value != "error_no_prediction"]
        err_2d = np.mean(err_filtered_2d)
        rob_3d = self.get_robustness_score(self.rob_frames_counter_3d)
        err_filtered_3d = [value for value in self.err_3d if value != "error_negative_disparity" and value != "error_no_prediction"]
        err_3d = np.mean(err_filtered_3d)
        n_f_2d = len(self.iou_list)
        n_f_rob = self.n_visible_and_not_diff + self.n_excessive_frames
        n_f_3d = len(err_filtered_3d)
        stats_anchor.acc = acc
        stats_anchor.rob_2d = rob_2d
        stats_anchor.err_2d = err_2d
        stats_anchor.rob_3d = rob_3d
        stats_anchor.err_3d = err_3d
        stats_anchor.n_f_2d = n_f_2d
        stats_anchor.n_f_rob = n_f_rob
        stats_anchor.n_f_3d = n_f_3d


def get_bbox_corners(bbox):
    top_left = (bbox[0], bbox[1])
    bot_right = (bbox[0] + bbox[2], bbox[1] + bbox[3])
    return top_left, bot_right


def draw_bb_in_frame(im1, im2, bbox1_gt, bbox1_p, bbox2_gt, bbox2_p, is_difficult, thck):
    color_gt = (0, 255, 0)  # Green (If the ground-truth is used to assess)
    color_p = (255, 0, 0)  # Blue (Prediction always shown in Blue)
    if is_difficult:
        color_gt = (0, 215, 255) # Orange (If the ground-truth is NOT used to assess)
    # Ground-truth
    if bbox1_gt is not None:
        top_left, bot_right = get_bbox_corners(bbox1_gt)
        im1 = cv.rectangle(im1, top_left, bot_right, color_gt, thck)
    if bbox2_gt is not None:
        top_left, bot_right = get_bbox_corners(bbox2_gt)
        im2 = cv.rectangle(im2, top_left, bot_right, color_gt, thck)
    # Predicted
    if bbox1_p is not None:
        top_left, bot_right = get_bbox_corners(bbox1_p)
        im1 = cv.rectangle(im1, top_left, bot_right, color_p, thck)
    if bbox2_p is not None:
        top_left, bot_right = get_bbox_corners(bbox2_p)
        im2 = cv.rectangle(im2, top_left, bot_right, color_p, thck)
    im_hstack = np.hstack((im1, im2))
    return im_hstack


def assess_anchor(v, anch, ar, kss, is_visualization_off):
    # Create window for results animation
    if not is_visualization_off:
        # Parameters for visualization only! These parameters do not affect the results!
        window_name = "Assessment animation"
        thick = 2
        bbox1_p, bbox2_p = None, None
        cv.namedWindow(window_name, cv.WINDOW_KEEPRATIO)

    # Use video and load a specific key point
    t = None
    is_track_fail_2d = False
    is_track_fail_3d = False
    while v.cap.isOpened():
        v.frame_counter += 1 # Since it is initialized to -1, the first frame will be 0
        if is_track_fail_2d and is_track_fail_3d:
            if v.frame_counter > kss.TERMINATOR_FRAME:
                break
        else:
            # Get data of new frame
            frame = v.get_frame()
            if frame is None:
                break # frame = None when video reached the last frame
            im1, im2 = v.split_frame(frame)
        if v.frame_counter < anch:
            continue # skip frames before anchor

        bbox1_gt, bbox2_gt, is_difficult, is_visible_in_both_stereo = v.get_bbox_gt(v.frame_counter)

        if t is None:
            # Initialise the tracker
            if is_visible_in_both_stereo:
                if not is_difficult:
                    if v.is_bbox_inside_image(bbox1_gt, bbox2_gt):
                        t = Tracker(im1, im2, bbox1_gt, bbox2_gt) # Initialise
                        kss.add_ss() # Add a new sub-sequence when tracker is initialized
            continue

        if is_difficult or not is_visible_in_both_stereo:
            # add `ignore` to sub-sequence
            kss.add_iou_score("ignore", v.frame_counter)
        else:
            ar.n_visible_and_not_diff += 1
            if is_track_fail_2d:
                kss.add_iou_score(0, v.frame_counter)

        if not is_track_fail_2d or not is_track_fail_3d:
            # Update the tracker
            bbox1_p, bbox2_p = t.tracker_update(im1, im2)
            if is_difficult:
                # If `is_difficult` then the metrics are not be affected
                continue
            elif not is_visible_in_both_stereo:
                # If it is not visible, the robustness score can still be affected
                if (bbox1_p is not None and bbox1_gt is None) or\
                   (bbox2_p is not None and bbox2_gt is None):
                    # If the tracker made a prediction when the target is not visible
                    ar.n_excessive_frames += 1
            else:
                flag_track_fail_2d, flag_track_fail_3d, iou = ar.calculate_bbox_metrics(bbox1_gt, bbox1_p, bbox2_gt, bbox2_p, is_track_fail_2d, is_track_fail_3d)
                kss.add_iou_score(iou, v.frame_counter)
                if flag_track_fail_2d:
                    ar.use_scores_before_failure_2d()
                    is_track_fail_2d = True
                if flag_track_fail_3d:
                    ar.use_scores_before_failure_3d()
                    is_track_fail_3d = True


        # Show animation of the tracker
        is_track_fail_both = is_track_fail_2d or flag_track_fail_3d
        if not is_visualization_off and not is_track_fail_both:
            frame_aug = draw_bb_in_frame(im1, im2,
                                         bbox1_gt, bbox1_p,
                                         bbox2_gt, bbox2_p,
                                         is_difficult,
                                         thick)
            cv.imshow(window_name, frame_aug)
            cv.waitKey(1)


def print_results(str_start, stats):
    print("{} Acc:{:.3f} Rob_2D:{:.3f} Err_2D:{:.1f} [pixels] | Rob_3D:{:.3f} Err_3D:{:.2f} [mm]".format(str_start,
                                                                                                         stats.acc,
                                                                                                         stats.rob_2d,
                                                                                                         stats.err_2d,
                                                                                                         stats.rob_3d,
                                                                                                         stats.err_3d))


def assess_keypoint(v, kpt_anchors, kss, stats_kpt, config_results, is_visualization_off):
    """
        The keypoints are assessed throughout each video multiple times.
        Each time, the tracker is initialized at a different anchor points.
    """
    for anch_id, anch in enumerate(kpt_anchors):
        ar = AnchorResults(config_results["n_misses_allowed"],
                           config_results["iou_threshold"],
                           config_results["err_3d_threshold"],
                           v.Q)
        assess_anchor(v, anch, ar, kss, is_visualization_off)
        stats_anchor = Statistics() # To support multiple keypoints
        ar.get_full_metric(stats_anchor)
        print_results("\t\t\tAnchor {}, ".format(anch_id), stats_anchor)
        stats_kpt.append_stats(stats_anchor)
        # Re-start video for next anchor, or for next keypoint
        v.video_restart()
    assert(len(stats_kpt.acc) == len(kpt_anchors)) # Check that we have a acc list for each anchor
    stats_kpt.merge_stats()


def calculate_results_for_video(rank, stats_video, anchors, case_sample_path, is_to_rectify, config_results, is_visualization_off):
    # Load video
    v = Video(case_sample_path, is_to_rectify)
    # Iterate through each keypoint (each keypoint that was labelled throughout a video)
    for kpt_id in range(v.n_keypoints):
        kpt_anchors = anchors[kpt_id]
        # Load ground-truth for the specific keypoint being tested
        v.load_ground_truth(kpt_id)
        # After loading gt, get termination frame for that specific keypoint and video
        terminator_frame = v.get_terminator_frame()
        kss = KptSubSequences(terminator_frame, case_sample_path, kpt_id)
        stats_kpt = Statistics() # Save score for a kpt, and all its anchors
        assess_keypoint(v, kpt_anchors, kss, stats_kpt, config_results, is_visualization_off)
        stats_video.append_stats(stats_kpt)
        rank.add_kpt_ss(kss)
    # Check that we have statistics for each of the keypoints
    assert(len(stats_video.acc) == v.n_keypoints)
    stats_video.merge_stats()
    # Stop video after assessing all the keypoints of that specific video
    v.stop_video()


def calculate_results(config, valid_or_test, is_visualization_off):
    config_results = config["results"]
    is_to_rectify = config["is_to_rectify"]
    config_data = config[valid_or_test]

    rank = EAO_Rank(config_data["N_min"], config_data["N_max"])
    stats_case_all = Statistics() # For ALL cases

    if config_data["is_to_evaluate"]:
        print('{} dataset'.format(valid_or_test).upper())
        cases = utils.get_cases(config_data)
        # Go through each case
        for case in cases:
            stats_case = Statistics() # Statistics for a specific case
            print("\n\t{}".format(case.case_id))
            # Go through case sample (in other words, each video)
            for cs in case.case_samples:
                stats_video = Statistics() # Statistics for a video of a case (specifically, for all the keypoints of a video)
                calculate_results_for_video(rank, stats_video, cs.anchors, cs.case_sample_path, is_to_rectify, config_results, is_visualization_off)
                print_results("\t\t{}".format(cs.case_sample_path), stats_video)
                stats_case.append_stats(stats_video)
            # Get results for all the videos in the case
            stats_case.merge_stats()
            print_results("\t\tWeighted average, ", stats_case)
            stats_case_all.append_stats(stats_case)
        # Calculate the statistics for all cases together
        stats_case_all.merge_stats()
        print('{} final score:'.format(valid_or_test).upper())
        #rank.calculate_N_min_and_N_max() # Used by callenge organizers to get N_min and N_max for each dataset
        eao = rank.calculate_eao_score()
        print_results("\tEAO:{:.3f}".format(eao), stats_case_all)


# function called from `main.py`!
def evaluate_method(config, is_visualization_off):
    calculate_results(config, "validation", is_visualization_off)
    calculate_results(config, "test", is_visualization_off)
