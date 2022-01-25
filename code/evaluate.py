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
        self.bbox_counter = 0


    def load_ground_truth(self, ind_kpt):
        gt_data_path = os.path.join(self.case_sample_path, self.gt_files[ind_kpt])
        self.gt_data = utils.load_yaml_data(gt_data_path)


    def get_bbox_gt(self):
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
        bbxs = self.gt_data[self.bbox_counter]
        if bbxs is not None:
            bbox_1 = bbxs[0]
            bbox_2 = bbxs[1]
        self.bbox_counter += 1
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
                return frame
        self.cap.release()
        return None


    def stop_video(self):
        self.cap.release()


def draw_bb_in_frame(im1, im2, bbox1_gt, bbox2_gt, bbox1_p, bbox2_p, thck):
    color_gt = (0, 255, 0) # Green
    color_p  = (255, 0, 0) # Blue
    # Image left
    if bbox1_gt is not None:
        pt_top_left = (bbox1_gt[0], bbox1_gt[1])
        pt_bot_right = (bbox1_gt[0] + bbox1_gt[2], bbox1_gt[1] + bbox1_gt[3])
        im1 = cv.rectangle(im1, pt_top_left, pt_bot_right, color_gt, thck)
    if bbox1_p is not None:
        pt_top_left = (bbox1_p[0], bbox1_p[1])
        pt_bot_right = (bbox1_p[0] + bbox1_p[2], bbox1_p[1] + bbox1_p[3])
        im1 = cv.rectangle(im1, pt_top_left, pt_bot_right, color_p, thck)
    # Image right
    if bbox2_gt is not None:
        pt_top_left = (bbox2_gt[0], bbox2_gt[1])
        pt_bot_right = (bbox2_gt[0] + bbox2_gt[2], bbox2_gt[1] + bbox2_gt[3])
        im2 = cv.rectangle(im2, pt_top_left, pt_bot_right, color_gt, thck)
    if bbox2_p is not None:
        pt_top_left = (bbox2_p[0], bbox2_p[1])
        pt_bot_right = (bbox2_p[0] + bbox2_p[2], bbox2_p[1] + bbox2_p[3])
        im2 = cv.rectangle(im2, pt_top_left, pt_bot_right, color_p, thck)
    im_hstack = np.hstack((im1, im2))
    return im_hstack


def calculate_results_for_video(case_sample_path, is_to_rectify, valid_or_test):
    # Create window for results visualization
    cv.namedWindow(valid_or_test, cv.WINDOW_KEEPRATIO)
    thick = 2
    # Load video
    v = Video(case_sample_path, is_to_rectify)
    #""" # TODO: remove until here
    # Version 1 - Stop tracker when there is no ground-truth (for example due to occlusion)
    """
    t = None
    for ind_kpt in range(v.n_keypoints): # Iterate through all the keypoints
        # Load ground-truth for the specific keypoint being tested
        v.load_ground_truth(ind_kpt)
        while v.cap.isOpened():
            # Get data of new frame
            frame = v.get_frame()
            if frame is None:
                break
            im1, im2 = v.split_frame(frame)
            bbox1_gt, bbox2_gt = v.get_bbox_gt()
            bbox1_p, bbox2_p = None, None # For the visual animation
            # If is no bounding-boxes info then stop the tracker
            if bbox1_gt is None or bbox2_gt is None:
                t = None
            else:
                # Otherwise use the bounding-boxes info to re-start or update the tracker
                if t is None:
                    t = Tracker(im1, im2, bbox1_gt, bbox2_gt) # restart the tracker
                else:
                    bbox1_p, bbox2_p = t.tracker_update(im1, im2)
                    # Show animation of the tracker
            frame_aug = draw_bb_in_frame(im1, im2, bbox1_gt, bbox2_gt, bbox1_p, bbox2_p, thick)
            cv.imshow(valid_or_test, frame_aug)
            cv.waitKey(10)
        # Re-start video for assessing the next keypoint
        v.video_restart()
    # Stop video after assessing all the keypoints of that specific video
    v.stop_video()
    """
    # Version 2 - Let the tracker go and re-start if it fails to come back to the original place
    t = None
    has_tracking_failed = False
    for ind_kpt in range(v.n_keypoints): # Iterate through all the keypoints
        # Load ground-truth for the specific keypoint being tested
        v.load_ground_truth(ind_kpt)
        while v.cap.isOpened():
            # Get data of new frame
            frame = v.get_frame()
            if frame is None:
                break
            im1, im2 = v.split_frame(frame)
            bbox1_gt, bbox2_gt = v.get_bbox_gt()
            bbox1_p, bbox2_p = None, None # For the visual animation
            # Otherwise use the bounding-boxes info to re-start or update the tracker
            if t is None or has_tracking_failed:
                if bbox1_gt is not None and bbox2_gt is not None:
                    t = Tracker(im1, im2, bbox1_gt, bbox2_gt) # restart the tracker
            else:
                bbox1_p, bbox2_p = t.tracker_update(im1, im2)
                # Show animation of the tracker
            frame_aug = draw_bb_in_frame(im1, im2, bbox1_gt, bbox2_gt, bbox1_p, bbox2_p, thick)
            cv.imshow(valid_or_test, frame_aug)
            cv.waitKey(10)
        # Re-start video for assessing the next keypoint
        v.video_restart()
    # Stop video after assessing all the keypoints of that specific video
    v.stop_video()


def calculate_results(config, valid_or_test):
    is_to_rectify = config["is_to_rectify"]
    config_data = config[valid_or_test]
    if config_data["is_to_evaluate"]:
        case_paths, _ = utils.get_case_paths_and_links(config_data)
        # Go through each video
        for case_sample_path in case_paths:
            calculate_results_for_video(case_sample_path, is_to_rectify, valid_or_test)


def evaluate_method(config):
    calculate_results(config, "validation")
    calculate_results(config, "test")
