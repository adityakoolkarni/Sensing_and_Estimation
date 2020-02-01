####################################################
# Gradient Descent and Logistic Regression
#
#Author: Aditya Kulkarni
#UC San Diego 
#Course : ECE 276A

####################################################



from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
from draw_contours import shape_detect
import matplotlib.image as mpimg

def grad_descent(alpha, train_data, train_label, cv_data, cv_label, M = 50):
    '''
    Computes gradient descent using batch gradient descent
    train_data is the image stack of dimension n * 3; n is number of training examples and one value per RGB or HSV
    test_data and cv_data also follow the same understanding
    Label is of size n x 1; Either 0:(Not red) or 1:(Red)
    M is number of Epochs
    alpha is a learning rate

    TBD: Want to use test_data now?
    What to be returned??
    '''
    
    cv_loss_log = [0 for epoch in range(M)]
    train_loss_log = [0 for epoch in range(M)]

    shape_train_data = train_data.shape
    shape_cv_data = train_data.shape
    size_ftr_vctr = train_data.shape[1] 
    #wghts = train_data[10,:].reshape((1,size_ftr_vctr))
    wghts = np.mean(train_data,axis=0).reshape((1,size_ftr_vctr)) + 1e-3
    wghts[0,0] = 0
    #wghts = np.zeros((1,size_ftr_vctr))
    #wghts = np.load('good_wghts_unbalanced.npy')
    wghts = np.zeros((1,size_ftr_vctr))

    print("Wrights",wghts)
    
    # Gradient Descent Algorithm
    
    good_wghts = wghts
    prev_loss = np.inf

    for epoch in tqdm(range(M)):
        #Training 
        #print("Input to Logistic Regression",(train_data @ wghts.T).shape)
        #print (((train_label[:,0].reshape(shape_train_data[0],1) - lgst_reg(train_data @ wghts.T))).T)
        grad_sum = ((train_label[:,0].reshape(shape_train_data[0],1) - lgst_reg(train_data @ wghts.T))).T @ train_data
        #print("Sum of errors",grad_sum.shape)
        #Weight update
        wghts = wghts + alpha * grad_sum
        
        #Train loss calculation
        #train_loss = 0
        #for train_img in range(shape_train_data[0]):
        #    a = wghts @ resize_train_data[train_img,:].T
        #    train_loss = train_loss -(np.log(lgst_reg(a)) if train_label[train_img,0] == 1 else np.log(1 - lgst_reg(a)))

        a = train_data @ wghts.T
        z = lgst_reg(a)
        #print("###################### Logistic of Train ######################## ",z)
        train_loss_arr = np.where(train_label == 1,np.log(lgst_reg(a)),np.log(1 - lgst_reg(a)))
#        print("Train Loss",train_loss_arr)
        #print("Train Loss ", train_loss)

        train_loss = -1*sum(train_loss_arr)
       # print("train loss",train_loss)
            
        train_loss_log[epoch] = train_loss / (shape_train_data[0] * 2) #Average over number of images and number of classes
        
        #CV loss calculation
        a = cv_data @ wghts.T
        z = lgst_reg(a)
        #print("Summed Input to LGST for CV",a)
        cv_loss_arr = np.where(cv_label == 1,np.log(lgst_reg(a)),np.log(1 - lgst_reg(a)))
        cv_loss = -1 * sum(cv_loss_arr)
        #print("CV Loss",cv_loss)
#        print("CV Loss size and ", cv_loss_arr.shape)
        #print("cv loss",cv_loss)
        cv_loss_log[epoch] = cv_loss / (shape_cv_data[0] * 2) #Average over number of images and number of classes
        if(cv_loss < prev_loss):
            #print("Weights getting better and the loss is ",cv_loss[0],prev_loss)
            prev_loss = cv_loss
            good_wghts = wghts
            print("Saving Good Weights and Loss is ",cv_loss_log[epoch], train_loss_log[epoch])
        else:
            print("Loss is increasing",cv_loss_log[epoch], train_loss_log[epoch])
            pass
        
    #np.save('good_wghts',good_wghts)
    np.save('good_wghts_unbalanced',good_wghts)
    #################      Accuracy Calculation ##################

    
    #print("Summed Input to LGST for CV",a)
    plt.plot(np.arange(M),train_loss_log,label='train')
    plt.plot(np.arange(M),cv_loss_log,label='cv')
    plt.legend()
    plt.show()


    del train_loss_log
    del cv_loss_log
    del train_data
    del train_label

    del cv_data
    del cv_label

def lgst_reg(a):
    '''
    Computes the logistic activation function
    
    '''
    import numpy as np
    p_c0 = 1 / (1 + np.exp(-a))
    return p_c0
