from scipy.io import loadmat
from PIL import Image as im
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from roipoly import RoiPoly
from roipoly import MultiRoi
import cv2
import os
from compute_batch_grad_descent import grad_descent

def read():
    img = loadmat('/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/trainset/matlab.mat')
    print(img.keys())
    print(img['B'][0])


def py_test():
    mask = np.load('masks.npy')
    plt.imshow(mask,cmap='gray')
    plt.show()
    #for i in range(1,2):
    #    fig = plt.figure(1)
    #    img = mpimg.imread('/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/trainset/'+str(4)+'.jpg')
    #    plt.figure(1)
    #    plt.subplot(3,1,1)
    #    canvas = np.zeros(img.shape)
    #    canvas[:,:,2] = img[:,:,2]
    #    print(np.max(img)-np.min(img))
    #    canvas = (canvas - np.mean(canvas)) / (np.max(canvas) - np.min(canvas))
    #    print(np.mean(canvas))
    #    plt.imshow(canvas)
    #    plt.ylabel("B Component")
    #    plt.subplot(3,1,2)
    #    canvas = np.zeros(img.shape)
    #    canvas[:,:,1] = img[:,:,1]
    #    canvas = (canvas - np.mean(canvas)) / (np.max(canvas) - np.min(canvas))
    #    plt.imshow(canvas)
    #    plt.ylabel("G Component")
    #    plt.subplot(3,1,3)
    #    canvas = np.zeros(img.shape)
    #    canvas[:,:,0] = img[:,:,0]
    #    canvas = (canvas - np.mean(canvas)) / (np.max(canvas) - np.min(canvas))
    #    plt.imshow(canvas)
    #    plt.ylabel("R Component")

    #    
    #    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #    plt.figure(2)
    #    plt.subplot(3,1,1)
    #    canvas = np.zeros(img.shape)
    #    canvas[:,:,0] = img[:,:,0]
    #    canvas = (canvas - np.mean(canvas)) / (np.max(canvas) - np.min(canvas))
    #    plt.imshow(canvas)
    #    plt.ylabel("V Component")
    #    plt.subplot(3,1,2)
    #    canvas = np.zeros(img.shape)
    #    canvas[:,:,1] = img[:,:,1]
    #    canvas = (canvas - np.mean(canvas)) / (np.max(canvas) - np.min(canvas))
    #    plt.imshow(canvas)
    #    plt.ylabel("S Component")
    #    plt.subplot(3,1,3)
    #    canvas = np.zeros(img.shape)
    #    canvas[:,:,2] = img[:,:,2]
    #    canvas = (canvas - np.mean(canvas)) / (np.max(canvas) - np.min(canvas))
    #    plt.imshow(canvas)
    #    plt.ylabel("H Component")

    #    plt.figure(3)
    #    plt.imshow(img)
    #    plt.show()


def grab_image_segment(num_img = 5):
    '''
    Grabs a part of the image which has red sign for labelling and annotating and saves the output as a numpy array
    '''
    
    path = '/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/trainset/' 
    image_path_list = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpg' in file:
                image_path_list.append(os.path.join(r,file))

    for image in range(num_img):
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
        mask_complement = np.logical_not(mask_np)
        #img_masked = np.zeros(img.shape)
        #img_comp_masked = np.zeros(img.shape)
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

        #img_masked[:,:,0] =  img[:,:,0] * mask_np
        #img_masked[:,:,1] =  img[:,:,1] * mask_np
        #img_masked[:,:,2] =  img[:,:,2] * mask_np

        #img_comp_masked[:,:,0] =  img[:,:,0] * mask_complement
        #img_comp_masked[:,:,1] =  img[:,:,1] * mask_complement
        #img_comp_masked[:,:,2] =  img[:,:,2] * mask_complement

        plt.imshow(mask_np,cmap='gray')
        plt.show()

        red_ftr = np.zeros((mask_cntr,10))
        for data in range(mask_cntr):
            red_ftr[data,0] = 1
            red_ftr[data,1] = img_masked[data,0] ** 2                   
            red_ftr[data,2] = img_masked[data,1] ** 2
            red_ftr[data,3] = img_masked[data,2] ** 2
            red_ftr[data,4] = img_masked[data,0] *  img_masked[data,2] 
            red_ftr[data,5] = img_masked[data,1] *  img_masked[data,2] 
            red_ftr[data,6] = img_masked[data,1] *  img_masked[data,0] 
            red_ftr[data,7] = img_masked[data,0]                         
            red_ftr[data,8] = img_masked[data,1]                         
            red_ftr[data,9] = img_masked[data,2]                         

        non_red_ftr = np.zeros((mask_complement_cntr,10))
        for data in range(mask_cntr):
            non_red_ftr[data,0] = 1
            non_red_ftr[data,1] = img_comp_masked[data,0] ** 2                   
            non_red_ftr[data,2] = img_comp_masked[data,1] ** 2
            non_red_ftr[data,3] = img_comp_masked[data,2] ** 2
            non_red_ftr[data,4] = img_comp_masked[data,0] *  img_comp_masked[data,2] 
            non_red_ftr[data,5] = img_comp_masked[data,1] *  img_comp_masked[data,2] 
            non_red_ftr[data,6] = img_comp_masked[data,1] *  img_comp_masked[data,0] 
            non_red_ftr[data,7] = img_comp_masked[data,0]                         
            non_red_ftr[data,8] = img_comp_masked[data,1]                         
            non_red_ftr[data,9] = img_comp_masked[data,2]                         
   

        print("Red Data is {} Non Red Data is {}" .format(mask_cntr,mask_complement_cntr))
        np.save('image_masked_'+str(image),red_ftr)
        np.save('image_masked_complement_'+str(image),non_red_ftr)


def preprocess(num_train_img = 1,num_cv_img = 1, num_test_img = 1):
    '''
    Takes in masks and then generates training data set
    '''
    path = '/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/trainset/' 
    image_path_list = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpg' in file:
                image_path_list.append(os.path.join(r,file))

    path = '/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/' 
    mask_path_list = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.npy' in file:
                print(file)
                mask_path_list.append(os.path.join(r,file))


    mask_path_list.sort()
    train_data = None
    train_label = None
    for image in range(num_train_img):
        data_temp = np.load(mask_path_list[image]) / 255
        print(mask_path_list[image])
        train_data = data_temp if train_data is None else np.vstack((train_data,data_temp)) 
        if 'comple' in mask_path_list[image]:
            print('Hey problem')
            train_label = np.zeros((data_temp.shape[0],1)) if train_label is None else np.vstack((train_label, np.zeros((data_temp.shape[0],1))))
        else:
            print('Hey no problem')
            train_label = np.ones((data_temp.shape[0],1)) if train_label is None else np.vstack((train_label, np.ones((data_temp.shape[0],1))))


    cv_data = None
    cv_label = None
    for image in range(num_train_img,num_cv_img + num_train_img):
        data_temp = np.load(mask_path_list[image]) / 255
        cv_data = data_temp if cv_data is None else np.vstack((cv_data,data_temp)) 
        if 'comple' in mask_path_list[image]:
            cv_label = np.zeros((data_temp.shape[0],1)) if cv_label is None else np.vstack((cv_label, np.zeros((data_temp.shape[0],1))))
        else:
            cv_label = np.ones((data_temp.shape[0],1)) if cv_label is None else np.vstack((cv_label, np.ones((data_temp.shape[0],1))))

    test_date = None
    test_label = None
    for image in range(num_cv_img + num_train_img,num_cv_img + num_train_img + num_test_img):
        data_temp = np.load(mask_path_list[image]) / 255
        test_date = data_temp if test_date is None else np.vstack((test_date,data_temp)) 
        if 'comple' in mask_path_list[image]:
            test_label = np.zeros((data_temp.shape[0],1)) if test_label is None else np.vstack((test_label, np.zeros((data_temp.shape[0],1))))
        else:
            print("ALL RED")
            test_label = np.ones((data_temp.shape[0],1)) if test_label is None else np.vstack((test_label, np.ones((data_temp.shape[0],1))))



    assert train_data.shape[0] == train_label.shape[0]
    assert cv_data.shape[0] == cv_label.shape[0]
    assert test_date.shape[0] == test_label.shape[0]
    return train_data, train_label, cv_data, cv_label, test_date, test_label

    #for image in range(num_img):
    #    mask = np.load(mask_path_list[image]) / 255
    #    plt.imshow(mask)
    #    plt.show()



if __name__ == '__main__':
    #py_test()
    grab_image_segment(6)
    train_data, train_label, cv_data, cv_label, test_data, test_label= preprocess(4,1,1)
    grad_descent(1e-6, train_data, train_label, cv_data, cv_label, test_data, test_label, M = 50)

