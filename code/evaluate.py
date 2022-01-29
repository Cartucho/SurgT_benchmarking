import os
from code import utils
from code.method import Tracker

import cv2 as cv
import numpy as np

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
        bbxs = self.gt_data[frame_counter]
        if bbxs is not None:
            bbox_1 = bbxs[0]
            bbox_2 = bbxs[1]
        return bbox_1, bbox_2


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


class EAORank:
    def __init__(self, config):
        self.N_high = config["N_high"]
        self.N_low = config["N_low"]
        self.padded_list = []
        self.largest_vec = 0


    def append_padded_vector(self, padded_vec):
        self.padded_list.append(padded_vec)
        if len(padded_vec) > self.largest_vec:
            self.largest_vec = len(padded_vec)


    def calculate_eao_curve(self):
        self.phi = []
        print(self.padded_list)
        exit()
        #for i in range(self.largest_vec):
        #    Compute phi for each i 


    def calculate_eao_score(self):
        self.calculate_eao_curve()
        return None # TODO
        

class SSeq:
    def __init__(self):
        # Initalized once per video
        self.start_sub_sequence = 0  # frame count for the start of every ss
        self.sub_sequence_current = []  # all successful tracking vectors within a sub sequence
        self.accumulate_ss_accuracy = []  # appends the current accuracy score to the current succ track vector



class KptResults:
    def __init__(self, config):
        self.dir = config["dir"]
        self.n_misses_allowed = config["n_misses_allowed"]
        self.iou_threshold = config["iou_threshold"]
        self.accuracy_list = []
        self.precision_list = []
        self.robustness_frames_counter = 0
        self.excessive_frames_counter = 0
        self.n_visible = 0
        self.n_misses_successive = 0


    def reset_n_successive_misses(self):
        self.n_misses_successive = 0


    def assess_bbox_accuracy(self, bbox1_gt, bbox1_p, bbox2_gt, bbox2_p):
        """
        Check if stereo tracking is a success or not
        """
        if bbox1_gt is None or bbox2_gt is None:
            # collect in here
            if bbox1_p is not None or bbox2_p is not None:
                # If the tracker made a prediction when the target is not visible
                self.excessive_frames_counter += 1
            return False, None

        self.n_visible += 1

        left_accuracy = 0.0
        right_accuracy = 0.0
        if bbox1_p is not None and bbox2_p is not None:
            left_accuracy = self.get_accuracy_frame(bbox1_gt, bbox1_p)
            right_accuracy = self.get_accuracy_frame(bbox2_gt, bbox2_p)

        self.accuracy_list.append([left_accuracy, right_accuracy])

        if left_accuracy > self.iou_threshold and right_accuracy > self.iou_threshold:
            self.robustness_frames_counter += 1
            left_precision = self.get_precision_centroid_frame(bbox1_gt, bbox1_p)
            right_precision = self.get_precision_centroid_frame(bbox2_gt, bbox2_p)
            self.precision_list.append([left_precision, right_precision])
            self.reset_n_successive_misses()
            return False, np.mean([left_accuracy, right_accuracy])

        self.n_misses_successive += 1
        if self.n_misses_successive > self.n_misses_allowed:
            # Keep only the accuracies before tracking failure
            del self.accuracy_list[-self.n_misses_successive:]
            self.reset_n_successive_misses()
            return True, np.mean([left_accuracy, right_accuracy])
        return False, np.mean([left_accuracy, right_accuracy])


    def get_accuracy_frame(self, bbox_gt, bbox_p):
        # TODO: here we could have used the function get_bbox_corners(), to get x1 ....
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

    def get_precision_centroid_frame(self, bbox_gt, bbox_p):
        cp_gt = np.array([bbox_gt[0]+bbox_gt[2]/2., bbox_gt[1]+bbox_gt[3]/2.])
        cp_p = np.array([bbox_p[0]+bbox_p[2]/2., bbox_p[1]+bbox_p[3]/2.])
        diag_gt = np.sqrt(np.sum(bbox_gt[2]**2+bbox_gt[3]**2))/2 # /2 because maximum overlap len
        diag_p = np.sqrt(np.sum(bbox_p[2]**2+bbox_p[3]**2))/2
        cp_distance = 1 - np.sqrt(np.sum(np.abs(cp_gt-cp_p)**2))/(diag_gt+diag_p)
        assert(cp_distance >= 0.0 and cp_distance <= 1.0)
        return cp_distance

    def get_full_metric(self):
        """
        Only happens after all frames are processed, end of video for loop!
        """
        # HOW SHOULD WE SEPARATE LEFT AND RIGHT
        acc = np.mean([np.sum(np.array(self.accuracy_list)[:, 0]) / self.n_visible, np.sum(np.array(self.accuracy_list)[:, 1]) / self.n_visible])
        prec = np.mean([np.sum(np.array(self.precision_list)[:, 0]) / self.n_visible, np.sum(np.array(self.precision_list)[:, 1]) / self.n_visible])
        rob = self.robustness_frames_counter / (self.n_visible + self.excessive_frames_counter)
        assert(rob >= 0.0 and rob <= 1.0)
        return acc, prec, rob # acc, prec, rob



def get_bbox_corners(bbox):
    top_left = (bbox[0], bbox[1])
    bot_right = (bbox[0] + bbox[2], bbox[1] + bbox[3])
    return top_left, bot_right


def draw_bb_in_frame(im1, im2, bbox1_gt, bbox2_gt, bbox1_p, bbox2_p, thck):
    color_gt = (0, 255, 0)  # Green
    color_p = (255, 0, 0)  # Blue
    # Image left
    if bbox1_gt is not None:
        top_left, bot_right = get_bbox_corners(bbox1_gt)
        im1 = cv.rectangle(im1, top_left, bot_right, color_gt, thck)
    if bbox1_p is not None:
        top_left, bot_right = get_bbox_corners(bbox1_p)
        im1 = cv.rectangle(im1, top_left, bot_right, color_p, thck)
    # Image right
    if bbox2_gt is not None:
        top_left, bot_right = get_bbox_corners(bbox2_gt)
        im2 = cv.rectangle(im2, top_left, bot_right, color_gt, thck)
    if bbox2_p is not None:
        top_left, bot_right = get_bbox_corners(bbox2_p)
        im2 = cv.rectangle(im2, top_left, bot_right, color_p, thck)
    im_hstack = np.hstack((im1, im2))
    return im_hstack


def assess_bbox(ss, rank, frame_counter, kr, bbox1_gt, bbox1_p, bbox2_gt, bbox2_p):
    # TODO: if hard, set GT to None
    if bbox1_gt is None or bbox2_gt is None:  # if GT is none, its the end of a ss
        if len(ss.sub_sequence_current) > 0:
            ss.sub_sequence_current.append(ss.accumulate_ss_accuracy)  # appends the final accuracy vector
            ss.accumulate_ss_accuracy = []
            ss.end_sub_sequence = frame_counter  # frame end of ss
            bias = 0  # start at end frame of previous vector
            for ss_tmp in ss.sub_sequence_current:
                pad_req = ss.end_sub_sequence-ss.start_sub_sequence-len(ss_tmp)-bias  # length of padding req
                rank.append_padded_vector(ss_tmp + [0.]*pad_req)  # padding and appending to list
                bias += len(ss_tmp)
            ss.sub_sequence_current = []
        ss.start_sub_sequence = frame_counter + 1
            
    reset_flag, accuracy_value = kr.assess_bbox_accuracy(bbox1_gt, bbox1_p, bbox2_gt, bbox2_p)
    if reset_flag:
        ss.sub_sequence_current.append(ss.accumulate_ss_accuracy)
        ss.accumulate_ss_accuracy = []
    else:
        if accuracy_value is not None:
            ss.accumulate_ss_accuracy.append(accuracy_value)
    return reset_flag


def assess_keypoint(rank, v, kr):
    # Create window for results animation
    window_name = "Assessment animation"  # TODO: hardcoded
    thick = 2  # TODO: hardcoded
    bbox1_p, bbox2_p = None, None # For the visual animation
    cv.namedWindow(window_name, cv.WINDOW_KEEPRATIO)

    # Variables for the assessment
    t = None
    ss = SSeq()

    # Use video and load a specific key point
    while v.cap.isOpened():
        # Get data of new frame
        frame, frame_counter = v.get_frame()
        if frame is None:
            break
        im1, im2 = v.split_frame(frame)
        bbox1_gt, bbox2_gt = v.get_bbox_gt(frame_counter)

        if t is None:
            # Initialise or re-initialize the tracker
            if bbox1_gt is not None and bbox2_gt is not None:
                # We can only initialize if we have ground-truth bboxes
                t = Tracker(im1, im2, bbox1_gt, bbox2_gt)
        else:
            # Update the tracker
            bbox1_p, bbox2_p = t.tracker_update(im1, im2)
            # Compute metrics for video and keep track of sub-sequences
            reset_flag = assess_bbox(ss, rank, frame_counter, kr, bbox1_gt, bbox1_p, bbox2_gt, bbox2_p)
            if reset_flag:
                # If the tracker failed then we need to set it to None so that we re-initialize
                t = None
                reset_flag = False
                # In visual animation, we hide the last predicted bboxs when the tracker fails
                bbox1_p, bbox2_p = None, None

        # Show animation of the tracker
        frame_aug = draw_bb_in_frame(im1, im2, bbox1_gt, bbox2_gt, bbox1_p, bbox2_p, thick)
        cv.imshow(window_name, frame_aug)
        cv.waitKey(1)

    # Do one last to finish the sub-sequences without changing the results
    assess_bbox(ss, rank, frame_counter, kr, None, None, None, None)


def calculate_results_for_video(rank,case_sample_path, is_to_rectify, config_results):
    # Load video
    v = Video(case_sample_path, is_to_rectify)

    # for when there are multiple keypoints
    keypoints_acc = []
    keypoints_prec = []
    keypoints_rob = []

    # Iterate through all the keypoints
    for ind_kpt in range(v.n_keypoints):
        # Load ground-truth for the specific keypoint being tested
        v.load_ground_truth(ind_kpt)
        kr = KptResults(config_results)
        assess_keypoint(rank, v, kr)
        acc, prec, rob = kr.get_full_metric()
        keypoints_acc.append(acc)
        keypoints_prec.append(prec)
        keypoints_rob.append(rob)
        # Re-start video for assessing the next keypoint
        v.video_restart()

    # Stop video after assessing all the keypoints of that specific video
    v.stop_video()

    return np.mean(keypoints_acc), np.mean(keypoints_prec), np.mean(keypoints_rob)


def calculate_results(config, valid_or_test):
    config_results = config["results"]
    rank = EAORank(config_results)
    is_to_rectify = config["is_to_rectify"]
    config_data = config[valid_or_test]
    if config_data["is_to_evaluate"]:
        case_paths, _ = utils.get_case_paths_and_links(config_data)
        # Go through each video
        dataset_acc = [] # TODO: this should be inside the rank class!
        dataset_prec = [] # TODO: this should be inside the rank class!
        dataset_rob = [] # TODO: this should be inside the rank class!
        counter = 0 # TODO: remove
        for case_sample_path in case_paths:
            if counter == 11:# TODO: remove
                acc, prec, rob = calculate_results_for_video(rank,case_sample_path, is_to_rectify, config_results)
                print("{} Acc:{} Prec:{} Rob:{}".format(case_sample_path, acc, prec, rob))

                dataset_acc.append(acc) # TODO: this should be inside the rank class!
                dataset_prec.append(prec) # TODO: this should be inside the rank class!
                dataset_rob.append(rob) # TODO: this should be inside the rank class!
            counter += 1 # TODO: remove

    eao = rank.calculate_eao_score()

    return eao, np.mean(dataset_acc), np.mean(dataset_prec), np.mean(dataset_rob)


def evaluate_method(config):

    print('VALIDATION DATASET')
    eao, acc, prec, rob = calculate_results(config, "validation")
    print('VALIDATION FINAL SCORE')
    print("EAO:{} Accuracy:{} Precision:{} Roustness:{}".format(eao, acc, prec, rob))

    #print('TEST DATASET')
    #calculate_results(config, "test")
