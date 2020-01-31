import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate

def draw_bounding_box(img,threshed_img):
    # read and scale down image
    # wget https://bigsnarf.files.wordpress.com/2017/05/hammer.png #black and white
    # wget https://i1.wp.com/images.hgmsites.net/hug/2011-volvo-s60_100323431_h.jpg
    #img = cv2.pyrDown(cv2.imread('2011-volvo-s60_100323431_h.jpg', cv2.IMREAD_UNCHANGED))
    
    # threshold image
    #_, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                   # 127, 255, cv2.THRESH_BINARY)
    # find contours and get the external one
    
    print(type(threshed_img[0,100]),threshed_img[0,100])
    threshed_img = threshed_img.astype(np.uint8)
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
    #                cv2.CHAIN_APPROX_SIMPLE)
    
    # with each contour, draw boundingRect in green
    # a minAreaRect in red and
    # a minEnclosingCircle in blue
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
        # get the min area rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        # draw a red 'nghien' rectangle
        cv2.drawContours(img, [box], 0, (0, 0, 255))
    
        # finally, get the min enclosing circle
        #(x, y), radius = cv2.minEnclosingCircle(c)
        # convert all values to int
        #center = (int(x), int(y))
        #radius = int(radius)
        # and draw the circle in blue
        #img = cv2.circle(img, center, radius, (255, 0, 0), 2)
    
    print(len(contours))
    cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
    
    cv2.imshow("contours", img)
    
    cv2.imshow("contours", img)
    
    while True:
        key = cv2.waitKey(1)
        if key == 27: #ESC key to break
            break
    
    cv2.destroyAllWindows()



def region_props(image):
    image = np.zeros((600, 600))
    
    rr, cc = ellipse(300, 350, 100, 220)
    print("Hey",rr,cc)
    image[rr, cc] = 1
    plt.imshow(image,cmap='gray')
    plt.show()
    
    image = rotate(image, angle=15, order=0)
    
    rr, cc = ellipse(100, 100, 60, 50)
    image[rr, cc] = 1
    
    label_img = label(image)
    regions = regionprops(label_img)

    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)
    
    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length
    
        ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        ax.plot(x0, y0, '.g', markersize=15)
    
        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=2.5)
    
    ax.axis((0, 600, 600, 0))
    plt.show()

def shape_detect(bw,path):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    from skimage import data
    from skimage.filters import threshold_otsu
    from skimage.segmentation import clear_border
    from skimage.measure import label, regionprops
    from skimage.morphology import closing, square
    from skimage.color import label2rgb
    import matplotlib.image as mpimg 
    
    
    # apply threshold

    image = mpimg.imread(path)
    # remove artifacts connected to image border
    cleared = clear_border(bw)
    
    # label image regions
    label_image = label(cleared)
    image_label_overlay = label2rgb(label_image, image=image)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)
    
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 2000:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            print("BOX Detected!!",minr, minc, maxr, maxc)
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
    
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


if '__main__' == __name__ : 
    shape_detect()
    #region_props()
