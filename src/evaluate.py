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
        if is_to_rectify:
            self.calib_path = os.path.join(case_sample_path, "calibration.yaml")
            utils.is_path_file(self.calib_path)
            self.load_calib_data()
            self.stereo_rectify()
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
        return bbox_1, bbox_2, is_difficult


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
                self.frame_counter += 1
                return frame, self.frame_counter
        self.cap.release()
        return None, self.frame_counter


    def stop_video(self):
        self.cap.release()


class Statistics:
    def __init__(self):
        self.acc_list = []
        self.rob_list = []
        self.err_2d_list = []
        self.err_3d_list = []


    def append_stats(self, acc, rob, err_2d, err_3d):
        self.acc_list.append(acc)
        self.rob_list.append(rob)
        self.err_2d_list.append(err_2d)
        self.err_3d_list.append(err_3d)


    def get_stats_mean(self):
        mean_acc = np.mean(self.acc_list)
        mean_rob = np.mean(self.rob_list)
        mean_err_2d = np.mean(self.err_2d_list)
        mean_err_3d = np.mean(self.err_3d_list)
        return mean_acc, mean_rob, mean_err_2d, mean_err_3d


class EAO_Rank:
    def __init__(self, N_min, N_max):
        self.all_padded_ss_list = []
        self.N_min = N_min
        self.N_max = N_max


    def append_ss_list(self, padded_list):
        self.all_padded_ss_list += padded_list


    def update_ss_length(self):
        if not self.all_padded_ss_list:
            return
        all_ss_len = []
        for ss in self.all_padded_ss_list:
            if ss:
                # If list not empty
                len_ss = len(ss)
                ss_copy = ss.copy()
                # Do not count with "is_difficult" at the tail of the list
                for i in range(len_ss):
                    val = ss_copy.pop()
                    if val != "is_difficult":
                        break
                all_ss_len.append(len_ss - i)
        all_ss_len = np.array(all_ss_len)
        self.ss_len_max = np.amax(all_ss_len)
        """ HOWTO Calculate N_min and N_max:
            Step 1. Uncomment break in `assess_bbox()` to not include reseted ss
            Step 2. Uncomment the next line `self.calculate_N_min_and_N_high`
        """
        #self.calculate_N_min_and_N_high(all_ss_len)


    def calculate_N_min_and_N_high(self, all_ss_len):
        ss_len_mean = np.mean(all_ss_len)
        ss_len_std = np.std(all_ss_len)
        N_min = max(1, int(round(ss_len_mean - ss_len_std)))
        N_max = int(round(ss_len_mean + ss_len_std))
        print("Mean:{} Std:{} N_min:{} N_max:{}".format(ss_len_mean,
                                                        ss_len_std,
                                                        N_min,
                                                        N_max))
        # Show histogram
        bins = int(self.ss_len_max / 10) # Make bars of 10 frames
        hist, bin_edges = np.histogram(all_ss_len, bins=bins)
        _ = plt.hist(all_ss_len, bins=bin_edges)  # arguments are passed to np.histogram
        plt.show()


    def calculate_eao_curve(self):
        self.eao_curve = []
        self.ss_len_max = 0
        self.update_ss_length()
        for i in range(self.ss_len_max):
            score = 0
            ss_sum = 0.0
            ss_counter = 0
            for ss in self.all_padded_ss_list:
                if len(ss) > i:
                    if ss[i] == "is_difficult":
                        continue
                    ss_sum += ss[i]
                    ss_counter += 1
            if ss_counter == 0:
                # This happens when all of the ss had the value "is_difficult" for frame i
                self.eao_curve.append("is_difficult")
                continue
            score = ss_sum / ss_counter
            self.eao_curve.append(score)


    def calculate_eao_score(self):
        self.calculate_eao_curve()
        if not self.eao_curve:
            # If empty list
            return 0.0
        eao_curve_N = self.eao_curve[self.N_min:self.N_max]
        # Remove any "is_difficult" score
        eao_curve_N_filt = [value for value in eao_curve_N if value != "is_difficult"]
        return np.mean(eao_curve_N_filt)
        

class SSeq:
    def __init__(self):
        # Initalized once per video
        self.start_sub_sequence = 0  # frame count for the start of every ss
        self.sub_sequence_current = []  # all successful tracking vectors within a sub sequence
        self.accumulate_ss_iou = []  # accumulates the IoU scores of the running tracker
        self.padded_list = []


    def append_padded_vector(self, padded_vec):
        self.padded_list.append(padded_vec)



class KptResults:
    def __init__(self, n_misses_allowed, iou_threshold, Q=None):
        self.n_misses_allowed = n_misses_allowed
        self.iou_threshold = iou_threshold
        self.Q = Q
        self.iou_list = []
        self.err_2d_list = []
        self.err_3d_list = []
        self.robustness_frames_counter = 0
        self.n_excessive_frames = 0
        self.n_visible = 0
        self.n_misses_successive = 0


    def reset_n_successive_misses(self):
        self.n_misses_successive = 0


    def calculate_bbox_metrics(self, bbox1_gt, bbox1_p, bbox2_gt, bbox2_p):
        """
        Check if stereo tracking is a success or not
        """
        if bbox1_gt is None or bbox2_gt is None:
            if bbox1_p is not None or bbox2_p is not None:
                # If the tracker made a prediction when the target is not visible
                self.n_excessive_frames += 1
            return False, None
        self.n_visible += 1

        iou = 0
        iou1 = 0
        iou2 = 0
        if bbox1_p is not None and bbox2_p is not None:
            iou1 = self.get_iou(bbox1_gt, bbox1_p)
            iou2 = self.get_iou(bbox2_gt, bbox2_p)
            # Use the mean overlap between the two images
            iou = np.mean([iou1, iou2])
        self.iou_list.append(iou)
        if iou1 > self.iou_threshold and iou2 > self.iou_threshold:
            self.robustness_frames_counter += 1
            self.calculate_l2_norm_errors(bbox1_gt, bbox1_p, bbox2_gt, bbox2_p)
            self.reset_n_successive_misses()
        # Otherwise it missed
        self.n_misses_successive += 1
        if self.n_misses_successive > self.n_misses_allowed:
            # Keep only the IoUs before tracking failure
            del self.iou_list[-self.n_misses_successive:]
            self.reset_n_successive_misses()
            return True, iou
        return False, iou


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


    def calculate_l2_norm_errors(self, bbox1_gt, bbox1_p, bbox2_gt, bbox2_p):
        centr_2d_gt_1 = self.get_bbox_centr(bbox1_gt)
        centr_2d_p_1 = self.get_bbox_centr(bbox1_p)
        centr_2d_gt_2 = self.get_bbox_centr(bbox2_gt)
        centr_2d_p_2 = self.get_bbox_centr(bbox2_p)
        # Get 2D error [pixels]
        err_2d_1 = self.get_l2_norm(centr_2d_gt_1, centr_2d_p_1)
        err_2d_2 = self.get_l2_norm(centr_2d_gt_2, centr_2d_p_2)
        err_2d = np.mean([err_2d_1, err_2d_2])
        self.err_2d_list.append(err_2d)
        # Get 3D error [mm]
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
            self.err_3d_list.append(err_3d)


    def get_iou(self, bbox_gt, bbox_p):
        x1, y1, x2, y2 = [bbox_gt[0], bbox_gt[1], bbox_gt[0]+bbox_gt[2], bbox_gt[1]+bbox_gt[3]]
        x3, y3, x4, y4 = [bbox_p[0], bbox_p[1], bbox_p[0]+bbox_p[2], bbox_p[1]+bbox_p[3]]
        x_inter1 = max(x1, x3)
        y_inter1 = max(y1, y3)
        x_inter2 = min(x2, x4)
        y_inter2 = min(y2, y4)
        widthinter = np.maximum(0,x_inter2 - x_inter1)
        heightinter = np.maximum(0,y_inter2 - y_inter1)
        areainter = widthinter * heightinter
        widthboxl = abs(x2 - x1)
        heightboxl = abs(y2 - y1)
        widthbox2 = abs(x4 - x3)
        heightbox2 = abs(y4 - y3)
        areaboxl = widthboxl * heightboxl
        areabox2 = widthbox2 * heightbox2
        areaunion = areaboxl + areabox2 - areainter
        iou = areainter / float(areaunion)
        assert(iou >= 0.0 and iou <= 1.0)
        return iou


    def get_accuracy_score(self):
        acc = 1.0
        if self.n_visible > 0:
            acc = np.sum(self.iou_list) / self.n_visible
        assert(acc >= 0.0 and acc <= 1.0)
        return acc


    def get_robustness_score(self):
        rob = 1.0
        denominator = self.n_visible + self.n_excessive_frames
        if denominator > 0:
            rob = self.robustness_frames_counter / denominator
        assert(rob >= 0.0 and rob <= 1.0)
        return rob

    def get_full_metric(self):
        """
        Only happens after all frames are processed, end of video for-loop!
        """
        acc = self.get_accuracy_score()
        rob = self.get_robustness_score()
        err_2d = np.mean(self.err_2d_list)
        err_3d = np.mean(self.err_2d_list)
        return acc, rob, err_2d, err_3d



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


def assess_bbox(ss, frame_counter, kr, bbox1_gt, bbox1_p, bbox2_gt, bbox2_p, is_difficult):
    if is_difficult:
        # If `is_difficult` then the metrics are not be affected
        ss.accumulate_ss_iou.append("is_difficult")
        return False

    if bbox1_gt is None or bbox2_gt is None:  # if GT is none, its the end of a ss
        if len(ss.sub_sequence_current) > 0:
            ss.sub_sequence_current.append(ss.accumulate_ss_iou)  # appends the final IoU vector
            ss.accumulate_ss_iou = []
            ss.end_sub_sequence = frame_counter  # frame end of ss
            bias = 0  # start at end frame of previous vector
            for ss_tmp in ss.sub_sequence_current:
                pad_req = ss.end_sub_sequence-ss.start_sub_sequence-len(ss_tmp)-bias  # length of padding req
                ss.append_padded_vector(ss_tmp + [0.] * pad_req)  # padding and appending to list
                bias += len(ss_tmp)
                #break # Add a break if you do NOT want to include the resetted ss
            ss.sub_sequence_current = []
        ss.start_sub_sequence = frame_counter + 1

    reset_flag, iou = kr.calculate_bbox_metrics(bbox1_gt, bbox1_p, bbox2_gt, bbox2_p)
    if reset_flag:
        ss.sub_sequence_current.append(ss.accumulate_ss_iou)
        ss.accumulate_ss_iou = []
    else:
        if iou is not None:
            ss.accumulate_ss_iou.append(iou)
    return reset_flag


def assess_keypoint(v, kr, ss):
    # Create window for results animation
    window_name = "Assessment animation"  # Name of the window. Does not affect the results!
    thick = 2  # thickness of bounding-box, for visualization only! Does not affect the results!
    bbox1_p, bbox2_p = None, None # For the visual animation
    cv.namedWindow(window_name, cv.WINDOW_KEEPRATIO)

    # Use video and load a specific key point
    t = None
    while v.cap.isOpened():
        # Get data of new frame
        frame, frame_counter = v.get_frame()
        if frame is None:
            break
        im1, im2 = v.split_frame(frame)
        bbox1_gt, bbox2_gt, is_difficult = v.get_bbox_gt(frame_counter)

        if t is None:
            # Initialise or re-initialize the tracker
            if bbox1_gt is not None and bbox2_gt is not None:
                # We can only initialize if we have ground-truth bboxes
                if not is_difficult:
                    # Only if bbox is not difficult to track
                    if v.is_bbox_inside_image(bbox1_gt, bbox2_gt):
                        # Only if the bbox is inside the image
                        t = Tracker(im1, im2, bbox1_gt, bbox2_gt)
        else:
            # Update the tracker
            bbox1_p, bbox2_p = t.tracker_update(im1, im2)
            # Compute metrics for video and keep track of sub-sequences
            reset_flag = assess_bbox(ss,
                                     frame_counter,
                                     kr,
                                     bbox1_gt, bbox1_p,
                                     bbox2_gt, bbox2_p,
                                     is_difficult)
            if reset_flag:
                # If the tracker failed then we need to set it to None so that we re-initialize
                t = None
                # In visual animation, we hide the last predicted bboxs when the tracker fails
                bbox1_p, bbox2_p = None, None

        # Show animation of the tracker
        frame_aug = draw_bb_in_frame(im1, im2,
                                     bbox1_gt, bbox1_p,
                                     bbox2_gt, bbox2_p,
                                     is_difficult,
                                     thick)
        cv.imshow(window_name, frame_aug)
        cv.waitKey(1)

    # Do one last to finish the sub-sequences without changing the results
    assess_bbox(ss, frame_counter, kr, None, None, None, None, False)


def calculate_results_for_video(rank, case_sample_path, is_to_rectify, config_results):
    # Load video
    v = Video(case_sample_path, is_to_rectify)

    # for when there are multiple keypoints
    stats = Statistics()

    # Iterate through all the keypoints
    for ind_kpt in range(v.n_keypoints):
        # Load ground-truth for the specific keypoint being tested
        v.load_ground_truth(ind_kpt)
        kr = KptResults(config_results["n_misses_allowed"],
                        config_results["iou_threshold"],
                        v.Q)
        ss = SSeq()
        assess_keypoint(v, kr, ss) # Assess a bounding-box throughout an entire video
        rank.append_ss_list(ss.padded_list)
        acc, rob, err_2d, err_3d = kr.get_full_metric()
        stats.append_stats(acc, rob, err_2d, err_3d)
        # Re-start video for assessing the next keypoint
        v.video_restart()

    # Check that we have statistics for each of the keypoints
    assert(len(stats.acc_list) == v.n_keypoints)

    # Stop video after assessing all the keypoints of that specific video
    v.stop_video()

    return stats.get_stats_mean()


def print_results(str_start, acc, rob, err_2d, err_3d):
    print("{} Acc:{:.3f} Rob:{:.3f} Err_2D: {:.1f} [pixels] Err_3D: {:.2f} [mm]".format(str_start,
                                                                                  acc,
                                                                                  rob,
                                                                                  err_2d,
                                                                                  err_3d))


def calculate_case_statitics(case_id, stats_case, stats_case_all):
    if case_id != -1:
        mean_acc, mean_rob, mean_err_2d, mean_err_3d = stats_case.get_stats_mean()
        print_results( "\tCase:{}".format(case_id), mean_acc, mean_rob, mean_err_2d, mean_err_3d)
        print("\n")
        # Append them to final statistics
        stats_case_all.append_stats(mean_acc, mean_rob, mean_err_2d, mean_err_3d)

    
def calculate_results(config, valid_or_test):
    config_results = config["results"]
    is_to_rectify = config["is_to_rectify"]
    config_data = config[valid_or_test]

    rank = EAO_Rank(config_data["N_min"], config_data["N_max"])
    case_id_prev = -1
    stats_case = Statistics() # For a specific case
    stats_case_all = Statistics() # For ALL cases

    if config_data["is_to_evaluate"]:
        print('{} dataset'.format(valid_or_test).upper())
        case_samples = utils.get_case_samples(config_data)
        # Go through each video
        for cs in case_samples:
            if cs.case_id != case_id_prev:
                # Flush previous video's resuts
                calculate_case_statitics(case_id_prev, stats_case, stats_case_all)
                stats_case = Statistics() # For a specific case
                case_id_prev = cs.case_id
            acc, rob, err_2d, err_3d = calculate_results_for_video(rank,
                                                                   cs.case_sample_path,
                                                                   is_to_rectify,
                                                                   config_results)
            print_results("\t\t{}".format(cs.case_sample_path), acc, rob, err_2d, err_3d)
            stats_case.append_stats(acc, rob, err_2d, err_3d)
        # Calculate statistics of the last case, at the end of for-loop
        calculate_case_statitics(cs.case_id, stats_case, stats_case_all)

        mean_acc, mean_rob, mean_err_2d, mean_err_3d = stats_case_all.get_stats_mean()
        print('{} final score:'.format(valid_or_test).upper())
        eao = rank.calculate_eao_score()
        print_results("\tEAO:{:.3f}".format(eao), mean_acc, mean_rob, mean_err_2d, mean_err_3d)


def evaluate_method(config):
    calculate_results(config, "validation")
    calculate_results(config, "test")
