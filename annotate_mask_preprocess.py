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
    
    path = '/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/best_images/' 
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

        #img_masked[:,:,0] =  img[:,:,0] * mask_np
        #img_masked[:,:,1] =  img[:,:,1] * mask_np
        #img_masked[:,:,2] =  img[:,:,2] * mask_np

        #img_comp_masked[:,:,0] =  img[:,:,0] * mask_complement
        #img_comp_masked[:,:,1] =  img[:,:,1] * mask_complement
        #img_comp_masked[:,:,2] =  img[:,:,2] * mask_complement


        

        #plt.imshow(mask_np,cmap='gray')
        #plt.show()

        print("Red Data is {} Non Red Data is {}" .format(mask_cntr,mask_complement_cntr))
        np.save('best_cln/image_masked_'+str(image),img_masked)
        np.save('best_cln/image_masked_complement_'+str(image),img_comp_masked)

        ############################### With Feature Calculation ####################################

        #red_ftr = np.zeros((mask_cntr,10))
        #for data in range(mask_cntr):
        #    red_ftr[data,0] = 1
        #    red_ftr[data,1] = img_masked[data,0] ** 2                   
        #    red_ftr[data,2] = img_masked[data,1] ** 2
        #    red_ftr[data,3] = img_masked[data,2] ** 2
        #    red_ftr[data,4] = img_masked[data,0] *  img_masked[data,2] 
        #    red_ftr[data,5] = img_masked[data,1] *  img_masked[data,2] 
        #    red_ftr[data,6] = img_masked[data,1] *  img_masked[data,0] 
        #    red_ftr[data,7] = img_masked[data,0]                         
        #    red_ftr[data,8] = img_masked[data,1]                         
        #    red_ftr[data,9] = img_masked[data,2]                         

        #non_red_ftr = np.zeros((mask_complement_cntr,10))
        #for data in range(mask_complement_cntr):
        #    non_red_ftr[data,0] = 1
        #    non_red_ftr[data,1] = img_comp_masked[data,0] ** 2                   
        #    non_red_ftr[data,2] = img_comp_masked[data,1] ** 2
        #    non_red_ftr[data,3] = img_comp_masked[data,2] ** 2
        #    non_red_ftr[data,4] = img_comp_masked[data,0] *  img_comp_masked[data,2] 
        #    non_red_ftr[data,5] = img_comp_masked[data,1] *  img_comp_masked[data,2] 
        #    non_red_ftr[data,6] = img_comp_masked[data,1] *  img_comp_masked[data,0] 
        #    non_red_ftr[data,7] = img_comp_masked[data,0]                         
        #    non_red_ftr[data,8] = img_comp_masked[data,1]                         
        #    non_red_ftr[data,9] = img_comp_masked[data,2]                         
   

        #print("Red Data is {} Non Red Data is {}" .format(mask_cntr,mask_complement_cntr))
        #np.save('image_masked_'+str(image),red_ftr)
        #np.save('image_masked_complement_'+str(image),non_red_ftr)

def get_data_and_label(mask_path_list,complement_mask_path_list,strt,end):
    '''
    Takes in RBG vecotrs and computes the feature vectors and returns
    A matrix of feature vectors and labels from start to end (excluding) image
    '''
    train_data = None
    train_label = None
    sanity_cnt = 1
    for image in range(strt,end):
        path = mask_path_list.pop()
        data_temp_mask = np.load(path) / 255
        if(data_temp_mask.shape[0] == 0):
            while(data_temp_mask.shape[0] == 0):
                path = mask_path_list.pop()
                data_temp_mask = np.load(path) / 255
                print("Faulty path??",path)

        path = complement_mask_path_list.pop()
        data_temp_comp_mask = np.load(path) / 255
        data_temp = np.vstack((data_temp_mask,data_temp_comp_mask))

        #train_label = np.zeros((data_temp.shape[0],1)) if train_label is None else np.vstack((train_label, np.zeros((data_temp.shape[0],1))))
        data_label_mask = np.ones((data_temp_mask.shape[0],1)) 
        data_label_comp_mask = np.zeros((data_temp_comp_mask.shape[0],1)) 
        data_label = np.vstack((data_label_mask,data_label_comp_mask))
        print("Train Label Shape",data_temp_mask.shape)
        assert data_label.shape[0] == data_temp.shape[0]


        ftr = np.zeros((data_temp.shape[0],10))
        #print("Current Directory",mask_path_list[image])
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

        train_data = ftr if train_data is None else np.vstack((train_data,ftr)) 
        #print("Train Data Shape",train_data)
        train_label = data_label if train_label is None else np.vstack((train_label,data_label)) 
        print("Train Label Concateneated",train_label)
    if (sanity_cnt == 0) :
        assert 1 == 2 

    #print(train_data.shape, train_label.shape)
    print('#################### Sum of ################',sum(train_label))

    assert train_data.shape[0] == train_label.shape[0]
    return train_data,train_label


#################################################################################
def separte_red_non_red(mask_path_list,complement_mask_path_list):
    '''
    Takes in RBG vecotrs and computes the feature vectors and returns
    A matrix of feature vectors and labels from start to end (excluding) image
    '''
    red_data = None
    non_red_data = None
    sanity_cnt = 1
    #for path in mask_path_list:
    #    #path = mask_path_list.pop()
    #    data_temp_mask = np.load(path) / 255
    #    if(data_temp_mask.shape[0] == 0):
    #        print("faulty path")
    #        continue
    #        #while(data_temp_mask.shape[0] == 0):
    #        #    continue
    #            #path = mask_path_list.pop()
    #            #data_temp_mask = np.load(path) / 255
    #            #print("Faulty path??",path)

    #    data_temp = data_temp_mask
    #    ftr = np.zeros((data_temp.shape[0],10))
    #    ftr[:,0] = np.ones(data_temp.shape[0])
    #    ftr[:,1] = data_temp[:,0] ** 2                   
    #    ftr[:,2] = data_temp[:,1] ** 2
    #    ftr[:,3] = data_temp[:,2] ** 2
    #    ftr[:,4] = data_temp[:,0] *  data_temp[:,2] 
    #    ftr[:,5] = data_temp[:,1] *  data_temp[:,2] 
    #    ftr[:,6] = data_temp[:,1] *  data_temp[:,0] 
    #    ftr[:,7] = data_temp[:,0]                         
    #    ftr[:,8] = data_temp[:,1]                         
    #    ftr[:,9] = data_temp[:,2]  

    #    red_data = ftr if red_data is None else np.vstack((red_data,ftr)) 
    #    #print("Train Data Shape",train_data)
    #    #print("Train Label Concateneated",train_label)

    #
    #np.save('red_data',red_data)
    #print("Number of Red samples ", red_data.shape[0])

    #exit(0)


    num_images = 0
    for path in complement_mask_path_list:
        #path = mask_path_list.pop()
        data_temp_mask = np.load(path) / 255
        if(data_temp_mask.shape[0] == 0):
            print("faulty path")
            continue

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
        if(num_images == 6):
            break
        num_images += 1
    

    print("Number of Non-Red samples ", non_red_data.shape[0])

    if (sanity_cnt == 0) :
        assert 1 == 2 


    np.save('non_red_data',non_red_data)

#################################################################################
def preprocess(num_train_img = 1,num_cv_img = 1, num_test_img = 1):
    '''
    Takes in masks and then generates training data set
    '''
    path = '/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/mask/' 
    mask_path_list = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.npy' in file:
                mask_path_list.append(os.path.join(r,file))


    path = '/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/complement_mask/' 
    complement_mask_path_list = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.npy' in file:
                complement_mask_path_list.append(os.path.join(r,file))

    #print("files",mask_path_list)
    #print("Number of images",num_train_img,num_cv_img,num_test_img)

    separte_red_non_red(mask_path_list,complement_mask_path_list)
    #train_data,train_label = get_data_and_label(mask_path_list,complement_mask_path_list,0,num_train_img)
    
    #cv_data, cv_label = get_data_and_label(mask_path_list,complement_mask_path_list,num_train_img,num_train_img+num_cv_img)


    ###############################  Old Testing Logic ######################################
    #for image in range(num_cv_img + num_train_img,num_cv_img + num_train_img + num_test_img):
    #    data_temp = np.load(mask_path_list[image]) / 255
    #    test_date = data_temp if test_date is None else np.vstack((test_date,data_temp)) 
    #    if 'comple' in mask_path_list[image]:
    #        test_label = np.zeros((data_temp.shape[0],1)) if test_label is None else np.vstack((test_label, np.zeros((data_temp.shape[0],1))))
    #    else:
    #        print("ALL RED")
    #        test_label = np.ones((data_temp.shape[0],1)) if test_label is None else np.vstack((test_label, np.ones((data_temp.shape[0],1))))

    #plt.imshow(test_img)
    #plt.show()
    #shffl_sctch_pd = np.hstack((test_data,test_label))
    #np.random.shuffle(shffl_sctch_pd) 

    #test_data = shffl_sctch_pd[:,0:-1].reshape(test_data.shape)
    #test_label = shffl_sctch_pd[:,-1].reshape(test_label.shape)

    #del shffl_sctch_pd


    ################################ New Testing ##############################


    #assert cv_data.shape[0] == cv_label.shape[0]
    ##assert test_ftr.shape[0] == test_label.shape[0]
    #return train_data, train_label, cv_data, cv_label

    #for image in range(num_img):
    #    mask = np.load(mask_path_list[image]) / 255
    #    plt.imshow(mask)
    #    plt.show()




if __name__ == '__main__':
    #py_test()
    #grab_image_segment(23)
    np.random.seed(42)
    #preprocess(num_train_img=6,num_cv_img=2,num_test_img=1)
    #train_data, train_label, cv_data, cv_label, test_data, test_label= preprocess(num_train_img=6,num_cv_img=2,num_test_img=1)
    #np.save('train_data',train_data)
    #np.save('train_label',train_label)
    #np.save('cv_data',cv_data)
    #np.save('cv_label',cv_label)
    #np.save('test_data',test_data)
    #np.save('test_label',test_label)


    #train_data = np.load('/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/train_data.npy')
    #train_label = np.load('/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/train_label.npy')
    #cv_data = np.load('/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/cv_data.npy')
    #cv_label = np.load('/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/cv_label.npy')
    #test_data = np.load('/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/test_data.npy')
    #test_label = np.load('/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/test_label.npy')

    #print("TTTTTTTTTTTTTTTTT and ", train_data.shape,sum(train_label))
    #print("TTTTTTTTTTTTTTTTT and ", np.where(train_label == 0))
    #print("CV and ", np.where(cv_label == 0))


    test_date = None
    test_label = None
    path = '/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/trainset/66.jpg' 
    test_img_raw = mpimg.imread(path) / 255
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

    
    #grad_descent(2e-3, train_data, train_label, cv_data, cv_label, test_ftr, test_label, path, M = 100)

    red_data = np.load('/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/red_data.npy')
    non_red_data = np.load('/home/aditya/Documents/Course_Work/sensing_and_estimation/HW_1/ECE276A_PR1/hw1_starter_code/non_red_data.npy')
    

    num_red_train = red_data.shape[0] - 5000
    num_non_red_train = num_red_train
    num_red_cv = 5000 
    num_non_red_cv = num_red_cv
    train_data = np.vstack((red_data[0:num_red_train],non_red_data[0:num_red_train]))
    train_label = np.vstack((np.ones((num_red_train,1)),np.zeros((num_red_train,1))))

    cv_data = np.vstack((red_data[num_red_train:num_red_train + num_red_cv],non_red_data[num_red_train: num_red_train+num_red_cv]))
    cv_label = np.vstack((np.ones((num_red_cv,1)),np.zeros((num_red_cv,1))))

    grad_descent(2e-5, train_data, train_label, cv_data, cv_label, test_ftr, test_label, path,M = 1000)

