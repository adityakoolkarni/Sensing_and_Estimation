'''
ECE276A WI20 HW1
Stop Sign Detector
'''

import os, cv2
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import matplotlib.image as mpimg 

import numpy as np
from compute_batch_grad_descent import grad_descent, lgst_reg
from skimage.color import rgb2hsv

class StopSignDetector():
    def __init__(self):
        '''
        	Initilize your stop sign detector with the attributes you need,
        	e.g., parameters of your classifier
        '''
    
        self.wghts_balanced = np.load('good_wghts.npy')
        self.wghts_unbalanced = np.load('good_wghts_unbalanced.npy')
        self.y_hght = None
    
    def segment_image(self, img):
        '''
        	Obtain a segmented image using a color classifier,
        	e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
        	call other functions in this class if needed
        	
        	Inputs:
        		img - original image
        	Outputs:
        		mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
        '''
        # YOUR CODE HERE
        b,g,r = cv2.split(img)       # get b,g,r
        rgb_img = cv2.merge([r,g,b])     # switch it to rgb
        print("Tpy on input",type(rgb_img))
        bounding_box, mask_img = self.test_logic(rgb_img)
        return bounding_box, mask_img

    def get_cordinates(self,box_cordinates,img_hght):
        '''
        Gets the cordinates for the left bottom and right top
        '''

        y_lt,x_lt,y_rb,x_rb = box_cordinates[0], box_cordinates[1], box_cordinates[2], box_cordinates[3]
        dist_x = x_rb - x_lt
        dist_y = y_lt - y_rb

        x_lb, y_lb = x_rb - dist_x, y_rb
        x_rt, y_rt = x_lt, y_rb + dist_y
        
        #Change co-ordinates for y 
        y_rt = img_hght - y_rt
        y_lb = img_hght - y_lb

        return [x_lb,y_lb,x_rt,y_rt]

    def shape_detect(self,bw,image):
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
    
        #image = mpimg.imread(path)
        # remove artifacts connected to image border
        cleared = clear_border(bw)
        
        # label image regions
        label_image = label(cleared)
        image_label_overlay = label2rgb(label_image, image=image)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(image_label_overlay)
        bounding_boxes = []
        
        for region in regionprops(label_image):
            # take regions with large enough areas
            if region.area >= 2000:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                print("BOX Detected!!",minr, minc, maxr, maxc)
                bounding_boxes.append(get_cordinates(rectangle,self.y_hght))
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
        
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()

        return bounding_boxes


    
    def test_logic(self,test_img_raw):
        '''
        This part tests the input images for stop signs
        '''
        test_date = None
        test_label = None
        #path = '/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/trainset/66.jpg' 
        #test_img_raw = mpimg.imread(path) / 255
        use_rgb = 0
        #test_img_raw[:,0],test_img_raw[:,1],test_img_raw[:,2] = test_img_raw[:,2],test_img_raw[:,1],test_img_raw[:,0] 
        test_img_rgb = test_img_raw
        self.y_hght = test_img_raw.shape[1]
        if(use_rgb == 0):
            test_img_raw = rgb2hsv(test_img_raw)
        #test_img_raw -= np.mean(test_img_raw,axis=0)
        #test_img_raw  /= np.std(test_img_raw,axis=0)
        print("TTTTTTTTTTTTTTT",test_img_raw.shape)
        test_ftr = np.zeros((test_img_raw.shape[0] * test_img_raw.shape[1],10))
        test_img = np.zeros((test_img_raw.shape[0] * test_img_raw.shape[1],3))
        test_img[:,0] = test_img_raw[:,:,0].flatten() 
        test_img[:,1] = test_img_raw[:,:,1].flatten() 
        test_img[:,2] = test_img_raw[:,:,2].flatten() 
        print("#######TEST IMAGE SHAPE",test_img.shape)
        test_ftr[:,0] = np.ones(test_img.shape[0])
        test_ftr[:,1] = test_img[:,0] ** 2                   
        test_ftr[:,2] = test_img[:,1] ** 2
        test_ftr[:,3] = test_img[:,2] ** 2
        test_ftr[:,4] = test_img[:,0] *  test_img[:,2] 
        test_ftr[:,5] = test_img[:,1] *  test_img[:,2] 
        test_ftr[:,6] = test_img[:,1] *  test_img[:,0] 
        test_ftr[:,7] = test_img[:,0]                         
        test_ftr[:,8] = test_img[:,1]                         
        test_ftr[:,9] = test_img[:,2]                         
    
        #plt.imshow(test_img_raw)
        #plt.show()
        test_label = test_img_raw.shape
        shape_test_data = test_ftr.shape
    
        pred_img = np.zeros((shape_test_data[0],1))
        a = test_ftr @ self.wghts_unbalanced.T
        pred_img = np.where(lgst_reg(a) > 0.5, np.ones((shape_test_data[0],1)), np.zeros((shape_test_data[0],1)))
        plt.figure()
        plt.subplot(2,2,1)
        est_img = pred_img.reshape(test_label[0],test_label[1])
        plt.imshow(est_img,cmap = 'gray')
    
        pred_img = np.zeros((shape_test_data[0],1))
        a = test_ftr @ self.wghts_balanced.T
        pred_img = np.where(lgst_reg(a) > 0.5, np.ones((shape_test_data[0],1)), np.zeros((shape_test_data[0],1)))

        plt.subplot(2,2,2)
        est_img = pred_img.reshape(test_label[0],test_label[1])
        plt.imshow(est_img,cmap = 'gray')

        plt.subplot(2,2,3)
        plt.imshow(test_img_rgb)

        plt.subplot(2,2,4)
        plt.imshow(test_img_rgb)
        plt.show()

        est_img_scaled = est_img * 255
        est_img_scaled.astype(np.uint8)
        bounding_boxes = self.shape_detect(est_img_scaled,test_img_rgb)
    
        return bounding_boxes, est_img


if __name__ == '__main__':
   folder = "trainset"
   my_detector = StopSignDetector()
   path = '/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/5.jpg'
   img = cv2.imread(path)
   bounding_box, mask_img = my_detector.segment_image(img)
   print("Bounding Bos is ", bounding_box)



   for filename in os.listdir(folder):
      # read one test image
      img = cv2.imread(os.path.join(folder,filename))
      cv2.imshow('image', img)
      cv2.waitKey(100)
      cv2.destroyAllWindows()
      bounding_box, mask_img = my_detector.segment_image(img)
      print("Bounding Bos is ", bounding_box)
   
   #Display results:
   #(1) Segmented images
   #	 mask_img = my_detector.segment_image(img)
   #(2) Stop sign bounding box
   #    boxes = my_detector.get_bounding_box(img)
   #The autograder checks your answers to the functions segment_image() and get_bounding_box()
   #Make sure your code runs as expected on the testset before submitting to Gradescope

