import os
from code import utils
from code.method import Tracker

import cv2 as cv
import numpy as np


"""
class BBox:
    def __init__(self, left, right, top, bottom):
        self.l = left
        self.r = right
        self.width = right - left
        self.centre_u = ((left + right) / 2.)
        self.t = top
        self.b = bottom
        self.height = bottom - top
        self.centre_v = ((top + bottom) / 2.)


    def get_bbox(self):
        return (self.l, self.t, self.width, self.height)


    def get_centre(self):
        return self.centre_u, self.centre_v
"""


class Video:
    def __init__(self, case_sample_path, is_to_rectify):
        # Load video info
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
        video_path = os.path.join(case_sample_path, name_video)
        #print(video_path)
        self.cap = cv.VideoCapture(video_path)
        self.frame_init = self.get_frame()


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
    thickness = 2
    # Load video
    v = Video(case_sample_path, is_to_rectify)
    if v.frame_init is not None:
        # Start Tracker
        im1, im2 = v.split_frame(v.frame_init)
        bbox1_gt = (200, 200, 100, 100)# TODO: remove
        bbox2_gt = (200, 200, 100, 100)# TODO: remove
        t = Tracker(im1, im2, bbox1_gt, bbox2_gt)
        # Loop through video / Go through each keypoint/bounding-box of a video
        while v.cap.isOpened():
            frame = v.get_frame()
            bbox1_gt = (200, 200, 100, 100)# TODO: remove
            bbox2_gt = (200, 200, 100, 100)# TODO: remove
            # When images are updated the tracker returns two updated bounding-boxes
            if frame is not None:
                im1, im2 = v.split_frame(frame)
                bbox1_p, bbox2_p = t.tracker_update(im1, im2)
                frame_augmented = draw_bb_in_frame(im1, im2, bbox1_gt, bbox2_gt, bbox1_p, bbox2_p, thickness)
                cv.imshow(valid_or_test, frame_augmented)
                cv.waitKey(10)
                # TODO Alistair: Compare predicted vs. ground-truth bounding-boxes


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
