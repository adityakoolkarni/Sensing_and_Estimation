from scipy.io import loadmat
from PIL import Image as im
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from roipoly import RoiPoly
from roipoly import MultiRoi
import skimage
import cv2
import os
from compute_batch_grad_descent import grad_descent, lgst_reg
from draw_contours import shape_detect


def grab_image_segment(num_img = 5):
    '''
    Grabs a part of the image which has red sign for labelling and annotating and saves the output as a numpy array
    '''
    
    #path = '/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/best_images/' 
    ## New Path ##
    path = '/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/image_for_label/'
    image_path_list = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpg' in file:
                image_path_list.append(os.path.join(r,file))

    print(image_path_list)
    for image in range(len(image_path_list)):
        if(image > 89):
            #img = mpimg.imread('/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/trainset/' + str(image) + '.jpg')
            img = mpimg.imread(image_path_list[image])
            #display the image for marking regions
            fig = plt.figure(1)
            plt.imshow(img)
            plt.show(block = False)
            #roi = RoiPoly(color='red',fig=fig)
            roi = MultiRoi(fig=fig)
            #plt.imshow(roi.get_mask(img))#,interpolation='nearest',cmap='Greys')
            img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            mask = None
            for msk in roi.rois.keys():
                mask = roi.rois[msk].get_mask(img_grey) if mask is None else np.logical_or(mask,roi.rois[msk].get_mask(img_grey))
            #mask = roi.get_mask(img_grey)
            mask_np = 1*mask
            mask_complement = 1*np.logical_not(mask_np)
            img_masked = np.zeros(img.shape)
            img_comp_masked = np.zeros(img.shape)
            mask_shape = sum(sum(mask_np))
            mask_complement_shape = img.shape[0]*img.shape[1] - mask_shape
            img_masked = np.zeros((mask_shape,3))
            img_comp_masked = np.zeros((mask_complement_shape,3))
            mask_cntr,mask_complement_cntr = 0,0
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if(mask[i,j]): #Area of interest
                        img_masked[mask_cntr,:] = img[i,j,:]
                        mask_cntr += 1
                    else:   
                        img_comp_masked[mask_complement_cntr,:] = img[i,j,:]
                        mask_complement_cntr += 1

            #plt.imshow(mask_np,cmap='gray')
            #plt.show()

            print("Red Data is {} Non Red Data is {}" .format(mask_cntr,mask_complement_cntr))
            print(image)
            ## Old Data Path ##
            #np.save('best_cln/image_masked_'+str(image),img_masked)
            #np.save('best_cln/image_masked_complement_'+str(image),img_comp_masked)

            np.save('labeled_images/red/red_'+str(image),img_masked)
            np.save('labeled_images/non_red/non_red_'+str(image),img_comp_masked)

#################################################################################
def separte_red_non_red(mask_path_list,complement_mask_path_list,use_rgb = 0):
    '''
    Takes in RBG vecotrs and computes the feature vectors and returns
    A matrix of feature vectors and labels from start to end (excluding) image
    '''
    red_data = None
    non_red_data = None
    sanity_cnt = 1
    for path in mask_path_list:
        #path = mask_path_list.pop()
        data_temp_mask = np.load(path) / 255
        if(data_temp_mask.shape[0] == 0):
            print("faulty path")
            continue
        elif(use_rgb == 0):
            data_temp_mask = rgb_to_hsv(data_temp_mask)
            #data_temp_mask -= np.mean(data_temp_mask,axis=0)
            #data_temp_mask /= np.std(data_temp_mask,axis=0)
        #else:
        #    data_temp_mask -= np.mean(data_temp_mask,axis=0)

        if(data_temp_mask.shape[0] != 0):
            data_temp = data_temp_mask
            ftr = np.zeros((data_temp.shape[0],10))
            ftr[:,0] = np.ones(data_temp.shape[0])
            ftr[:,1] = data_temp[:,0] ** 2                   
            ftr[:,2] = data_temp[:,1] ** 2
            ftr[:,3] = data_temp[:,2] ** 2
            ftr[:,4] = data_temp[:,0] *  data_temp[:,2] 
            ftr[:,5] = data_temp[:,1] *  data_temp[:,2] 
            ftr[:,6] = data_temp[:,1] *  data_temp[:,0] 
            ftr[:,7] = data_temp[:,0]                         
            ftr[:,8] = data_temp[:,1]                         
            ftr[:,9] = data_temp[:,2]  

            red_data = ftr if red_data is None else np.vstack((red_data,ftr)) 
        #print("Train Data Shape",train_data)
        #print("Train Label Concateneated",train_label)

    
    if(use_rgb == 1):
        np.save('red_data',red_data)
    else:
        np.save('red_data_hsv',red_data)
    print("Number of Red samples ", red_data.shape[0])

    del red_data

    num_images = 0
    for path in complement_mask_path_list:
        #path = mask_path_list.pop()
        data_temp_mask = np.load(path) / 255
        if(data_temp_mask.shape[0] == 0):
            print("faulty path")
            continue
        elif(use_rgb == 0):
            data_temp_mask = rgb_to_hsv(data_temp_mask)
            #data_temp_mask -= np.mean(data_temp_mask,axis=0)
            #data_temp_mask /= np.std(data_temp_mask,axis=0)
        #else:
        #    data_temp_mask -= np.mean(data_temp_mask,axis=0) #For RBG just do mean shifting
        
        if(data_temp_mask.shape[0] != 0):
            data_temp = data_temp_mask
            ftr = np.zeros((data_temp.shape[0],10))
            ftr[:,0] = np.ones(data_temp.shape[0])
            ftr[:,1] = data_temp[:,0] ** 2                   
            ftr[:,2] = data_temp[:,1] ** 2
            ftr[:,3] = data_temp[:,2] ** 2
            ftr[:,4] = data_temp[:,0] *  data_temp[:,2] 
            ftr[:,5] = data_temp[:,1] *  data_temp[:,2] 
            ftr[:,6] = data_temp[:,1] *  data_temp[:,0] 
            ftr[:,7] = data_temp[:,0]                         
            ftr[:,8] = data_temp[:,1]                         
            ftr[:,9] = data_temp[:,2]  

            non_red_data = ftr if non_red_data is None else np.vstack((non_red_data,ftr)) 
        #print("Train Data Shape",train_data)
        if(num_images == 13):
            break
        num_images += 1
    

    print("Number of Non-Red samples ", non_red_data.shape[0])


    if(use_rgb == 1):
        np.save('non_red_data',non_red_data)

    else:
        np.save('non_red_data_hsv',non_red_data)

    del non_red_data

from skimage.color import rgb2hsv
  
def rgb_to_hsv(data_rgb):
    '''
    Converts RGB to HSV 
    '''

    if(data_rgb.shape[0] % 2):
        data_temp = data_rgb[:-1,:].reshape(-1,2,3)
    else:
        data_temp = data_rgb.reshape(-1,2,3)
    data_hsv = rgb2hsv(data_temp)
    data_hsv = data_hsv.reshape(data_temp.shape[0] * data_temp.shape[1],3)

    return data_hsv




#################################################################################
def preprocess(num_train_img = 1,num_cv_img = 1, num_test_img = 1,use_rgb = 1):
    '''
    Takes in masks and then generates training data set
    '''
    #old path
    #path = '/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/mask/' 

    #New path
    path = '/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/labeled_images/red/'

    mask_path_list = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.npy' in file:
                mask_path_list.append(os.path.join(r,file))

    #old path
    #path = '/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/complement_mask/' 

    path = '/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/labeled_images/non_red/'

    complement_mask_path_list = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.npy' in file:
                complement_mask_path_list.append(os.path.join(r,file))

    
    separte_red_non_red(mask_path_list,complement_mask_path_list,use_rgb)


def test_logic(use_rgb,path):
    '''
    This part tests the input images for stop signs
    '''
    test_date = None
    test_label = None
    #path = '/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/trainset/66.jpg' 
    test_img_raw = mpimg.imread(path) / 255
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

    #good_wghts = np.load('good_wghts.npy')
    good_wghts = np.load('good_wghts_unbalanced.npy')
    pred = 0
    pred_img = np.zeros((shape_test_data[0],1))
    a = test_ftr @ good_wghts.T
    pred_img = np.where(lgst_reg(a) > 0.5, np.ones((shape_test_data[0],1)), np.zeros((shape_test_data[0],1)))
    plt.figure()
    plt.subplot(2,1,1)
    est_img = pred_img.reshape(test_label[0],test_label[1])
    plt.imshow(est_img,cmap = 'gray')

    plt.subplot(2,1,2)
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.show()
    est_img_scaled = est_img * 255
    est_img_scaled.astype(np.uint8)
    shape_detect(est_img_scaled,path)


if __name__ == '__main__':
    #py_test()
    #grab_image_segment(23)
    #exit(0)
    use_rgb = 1
    train = 0
    resave_data = False and (train == 1)
    alpha = 20e-5
    M = 50
    if(resave_data == True):
        preprocess(num_train_img=6,num_cv_img=2,num_test_img=1,use_rgb=use_rgb)

    if(use_rgb == 1):
        red_data = np.load('/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/red_data.npy')
        non_red_data = np.load('/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/non_red_data.npy')
    else:
        red_data = np.load('/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/red_data_hsv.npy')
        non_red_data = np.load('/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/non_red_data_hsv.npy')

    ###### Trianinig Part ########
    if(train == 1):
        if(use_rgb == 1):
            print("Begin training with RGB Learning Rate {} Number of Epochs {} ".format(alpha,M))
        else:
            print("Begin training with HSV Learning Rate {} Number of Epochs {} ".format(alpha,M))
            
        num_red_train = red_data.shape[0] - 2080000
        num_non_red_train = num_red_train + 100000 
        num_red_cv = 2080000
        num_non_red_cv = num_red_cv + 10000
        train_data = np.vstack((red_data[0:num_red_train],non_red_data[0:num_non_red_train]))
        train_label = np.vstack((np.ones((num_red_train,1)),np.zeros((num_non_red_train,1))))
        merge = np.hstack((train_data,train_label))
        np.random.shuffle(merge)
        train_data = merge[:,:-1].reshape(train_data.shape)
        train_label = merge[:,-1].reshape(train_label.shape)

        cv_data = np.vstack((red_data[num_red_train:num_red_train + num_red_cv],non_red_data[num_non_red_train: num_non_red_train+num_non_red_cv]))
        cv_label = np.vstack((np.ones((num_red_cv,1)),np.zeros((num_non_red_cv,1))))
        merge = np.hstack((cv_data,cv_label))

        cv_data = merge[:,:-1].reshape(cv_data.shape)
        cv_label = merge[:,-1].reshape(cv_label.shape)

        grad_descent(alpha, train_data, train_label, cv_data, cv_label,M)

    else:
        #path = '/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/best_images/' 
        path = '/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/trainset/' 
        for r, d, f in os.walk(path):
            for file in f:
                if '.jpg' in file:
                    print(file)
                    test_logic(use_rgb,path+str(file))

