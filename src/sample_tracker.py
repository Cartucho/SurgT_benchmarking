import cv2 as cv


class Tracker:
    """ This is a sample tracker, replace this class with your own tracker! """
    def __init__(self, init_im1, init_im2, init_bbox1, init_bbox2):
        self.t1 = cv.TrackerCSRT_create() # Track bbox on left image
        self.t2 = cv.TrackerCSRT_create() # Track bbox on right image
        self.t1.init(init_im1, init_bbox1)
        self.t2.init(init_im2, init_bbox2)


    def tracker_update(self, new_im1, new_im2):
        """
            Return two bboxes in format (u, v, width, height)

                                 (u,)   (u + width,)
                          (0,0)---.--------.---->
                            |
                       (,v) -     x--------.
                            |     |  bbox  |
              (,v + height) -     .________.
                            v
        """
        success1, bbox1 = self.t1.update(new_im1)
        success2, bbox2 = self.t2.update(new_im2)
        if not success1:
            bbox1 = None
        if not success2:
            bbox2 = None
        """
            In the ground-truth data, the 2D region of interest (inside the bboxes)
             is always visible in both the left and right image. If in any of
             the two images the tracking update fails, for example due to an
             occlusion in one of the images, then the tracker will be stopped.
             In other words, if `bbox1 == None` or `bbox2 == None`, the tracker stops.
             After the end of the occlusion, the evaluation code will automatically
             re-initialize the code (__init__()), and tracker_update will be called again.

            The other case where the tracker is re-initialized is if the evaluation
              code detects that the tracker is failing (bbox far away from the ground-truth one)
              for more than N successive frames.  
        """
        return bbox1, bbox2
