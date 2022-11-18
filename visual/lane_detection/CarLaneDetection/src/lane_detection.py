# from types import NoneType
import cv2
import numpy as np
import math
# from moviepy.editor import VideoFileClip
class lane_detection(object):
    def __init__(self):
        self.blur_ksize = 1  # Gaussian blur kernel size
        self.canny_lthreshold = 20  # Canny edge detection low threshold
        self.canny_hthreshold = 100  # Canny edge detection high threshold

        # Hough transform parameters
        self.rho = 1
        self.theta = np.pi / 180
        self.threshold = 30
        self.min_line_length = 30
        self.max_line_gap = 200
        self.slope = []

    def roi_mask(self, img, vertices):
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            mask_color = (255,) * channel_count
        else:
            mask_color = 255

        cv2.fillPoly(mask, vertices, mask_color)
        masked_img = cv2.bitwise_and(img, mask)
        return masked_img

    # def select_white_yellow(image):
    #     converted = image
    #     # white color mask
    #     lower = np.uint8([  0, 200,   0])
    #     upper = np.uint8([255, 255, 255])
    #     white_mask = cv2.inRange(converted, lower, upper)
    #     # yellow color mask
    #     # lower = np.uint8([ 10,   0, 100])
    #     # upper = np.uint8([ 40, 255, 255])
    #     # yellow_mask = cv2.inRange(converted, lower, upper)
    #     # # combine the mask
    #     # mask = cv2.bitwise_or(white_mask, yellow_mask)
    #     return white_mask#cv2.bitwise_and(image, image, mask = mask)

    def draw_roi(self, img, vertices):
        cv2.polylines(img, vertices, True, [255, 0, 0], thickness=2)


    def draw_lines(self, img, lines, color=[255, 0, 0], thickness=2):
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


    def hough_lines(self, img):
        lines = cv2.HoughLinesP(img, self.rho, self.theta, self.threshold, np.array([]), minLineLength=self.min_line_length,
                                maxLineGap=self.max_line_gap)
        
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        # print(str(type(lines)))
        if str(type(lines)) == "<class 'numpy.ndarray'>":
            self.draw_lanes(line_img, lines)
        return line_img


    def draw_lanes(self, img, lines, color=[255, 0, 0], thickness=8, threshold=20):
        left_lines, right_lines = [], []
        # former_data = []
        # if lines == []:
        #     pass
        # else:
        for line in lines:
            for x1, y1, x2, y2 in line:
                k = (y2 - y1) / (x2 - x1)
                if abs(k) < 0.5:
                    break
                else:
                    if k < 0:
                        left_lines.append(line)
                    else:
                        right_lines.append(line)
        # print(right_lines)
        if (len(left_lines) <= 0 or len(right_lines) <= 0):
            return img

        self.clean_lines(left_lines, 0.1)
        self.clean_lines(right_lines, 0.1)
        # self.slope.append((l_slope,r_slope))
        # if len(self.slope)>1:
        #     print(self.slope[-2][0],self.slope[-1][0])
        #     if self.slope[-2][0]-self.slope[-1][0] > threshold or self.slope[-2][1]-self.slope[-1][1] > threshold:
        #         return img
        #     self.slope = self.slope[:-1]
        left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]
        left_points = left_points + [(x2, y2) for line in left_lines for x1, y1, x2, y2 in line]
        right_points = [(x1, y1) for line in right_lines for x1, y1, x2, y2 in line]
        right_points = right_points + [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]
        
        left_vtx = self.calc_lane_vertices(left_points, 325, img.shape[0])
        right_vtx = self.calc_lane_vertices(right_points, 325, img.shape[0])

        cv2.line(img, left_vtx[0], left_vtx[1], color, thickness)
        cv2.line(img, right_vtx[0], right_vtx[1], color, thickness)
        


    # def former_slope(l_slope, r_slope):
    #     f_slope = []
        

    def clean_lines(self, lines, threshold):
        slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
        while len(lines) > 0:
            mean = np.mean(slope)
            diff = [abs(s - mean) for s in slope]
            idx = np.argmax(diff)
            if diff[idx] > threshold:
                slope.pop(idx)
                lines.pop(idx)
            else:
                break
        # return math.atan(slope[0])*180/math.pi


    def calc_lane_vertices(self, point_list, ymin, ymax):
        x = [p[0] for p in point_list]
        y = [p[1] for p in point_list]
        fit = np.polyfit(y, x, 1)
        fit_fn = np.poly1d(fit)

        xmin = int(fit_fn(ymin))
        xmax = int(fit_fn(ymax))

        return [(xmin, ymin), (xmax, ymax)]

    # def filter_region(image, vertices):
    #     """
    #     Create the mask using the vertices and apply it to the input image
    #     """
    #     mask = np.zeros_like(image)
    #     if len(mask.shape)==2:
    #         cv2.fillPoly(mask, vertices, 255)
    #     else:
    #         cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
    #     return cv2.bitwise_and(image, mask)

        
    # def select_region(image):
    #     """
    #     It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    #     """
    #     # first, define the polygon by vertices
    #     rows, cols = image.shape[:2]
    #     bottom_left  = [cols*0.1, rows*0.95]
    #     top_left     = [cols*0.4, rows*0.6]
    #     bottom_right = [cols*0.6, rows*0.95]
    #     top_right    = [cols*0.6, rows*0.6] 
    #     # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    #     vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    #     return filter_region(image, vertices)


def lane_detect(img):
    # print(type(img))
    blur_ksize = 1  # Gaussian blur kernel size
    canny_lthreshold = 30  # Canny edge detection low threshold
    canny_hthreshold = 100  # Canny edge detection high threshold

    # Hough transform parameters
    # self.rho = 1
    # self.theta = np.pi / 180
    # self.threshold = 40
    # self.min_line_length = 50
    # self.max_line_gap = 200
    # roi_vtx = np.array([[(0, img.shape[0]-70),  (450,430),(500,430),(img.shape[1], 600) ,(img.shape[1], img.shape[0]-70)]])
    roi_vtx = np.array([[(43, 596),  (1230, 596),
		(553, 410),
		(795, 410)]])

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # white_yellow = select_white_yellow(gray)
    # print(type(white_yellow))

    # lower_yellow = np.array([20, 100, 100], dtype = "uint8")
    # upper_yellow = np.array([30, 255, 255], dtype="uint8")

    # mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    # mask_white = cv2.inRange(gray, 200, 255)
    # mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    # mask_yw_image = cv2.bitwise_and(gray, mask_yw)

    blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    # blur_gray = cv2.GaussianBlur(mask_yw_image, blur_ksize)
    edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)
    lane = lane_detection()
    # cv2.imshow('1',edges)
    roi_edges = lane.roi_mask(edges, roi_vtx)
    cv2.imshow('',roi_edges)
    cv2.waitKey(50)
    # roi_edges = select_region(edges)
    line_img = lane.hough_lines(roi_edges)#, self.rho, self.theta, self.threshold, self.min_line_length, max_line_gap)
    # print(type(line_img))
    # img = np.array(img)
    # line_img = np.array(line_img)
    # img = img[:,:,:3]
    # print(img.shape)
    # print(line_img.shape)
    res_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)

    # plt.figure()
    # plt.imshow(gray, cmap='gray')
    # plt.savefig('../resources/gray.png', bbox_inches='tight')
    # plt.figure()
    # plt.imshow(blur_gray, cmap='gray')
    # plt.savefig('../resources/blur_gray.png', bbox_inches='tight')
    # plt.figure()
    # plt.imshow(edges, cmap='gray')
    # plt.savefig('../resources/edges.png', bbox_inches='tight')
    # plt.figure()
    # plt.imshow(roi_edges, cmap='gray')
    # plt.savefig('../resources/roi_edges.png', bbox_inches='tight')
    # plt.figure()
    # plt.imshow(line_img, cmap='gray')
    # plt.savefig('../resources/line_img.png', bbox_inches='tight')
    # plt.figure()
    # plt.imshow(res_img)
    # plt.savefig('../resources/res_img.png', bbox_inches='tight')
    # plt.show()


    return res_img, roi_edges


    # img = mplimg.imread("../resources/lane.jpg")
    # process_an_image(img)

    # output = '../resources/video_1_sol.mp4'
    # clip = VideoFileClip("../resources/video_1.mp4")
    # out_clip = clip.fl_image(process_an_image)
    # out_clip.write_videofile(output, audio=False)
