import cv2 as cv


class Tracker:
    """ This is a sample tracker, replace this class with your own tracker! """
    def __init__(self, init_im1, init_im2, init_bbox1, init_bbox2):
        self.t1 = cv.TrackerCSRT_create() # Track bbox on left image
        self.t2 = cv.TrackerCSRT_create() # Track bbox on right image
        self.t1.init(init_im1, init_bbox1)
        self.t2.init(init_im2, init_bbox2)


    def tracker_update(self, new_im1, new_im2):
        success1, bbox1 = self.t1.update(new_im1)
        if not success1:
            bbox1 = None
        success2, bbox2 = self.t2.update(new_im2)
        if not success2:
            bbox2 = None
        return bbox1, bbox2
