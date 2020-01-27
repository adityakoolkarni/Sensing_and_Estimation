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

def grad_descent(alpha, train_data, train_label, cv_data, cv_label, test_data, test_label, M = 50):
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
    shape_test_data = test_data.shape
    size_ftr_vctr = train_data.shape[1] 
    #wghts = np.zeros((1,size_ftr_vctr)) 
    wghts = train_data[0,:].reshape((1,size_ftr_vctr)) / 10000
    print("Wrights",wghts)
    
    # Gradient Descent Algorithm
    
    good_wghts = wghts
    prev_loss = np.inf
    for epoch in tqdm(range(M)):
        #Training 
        print((train_data @ wghts.T))
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
        train_loss_arr = np.where(train_label == 1,np.log(lgst_reg(a)),np.log(1 - lgst_reg(a)))

        train_loss = -1*sum(train_loss_arr)
        print("train loss",train_loss)
            
        train_loss_log[epoch] = train_loss / (shape_train_data[0] * 2) #Average over number of images and number of classes
        
        #CV loss calculation
        a = cv_data @ wghts.T
        z = lgst_reg(a)
        cv_loss_arr = np.where(cv_label == 1,np.log(lgst_reg(a)),np.log(1 - lgst_reg(a)))
        cv_loss = -1 * sum(cv_loss_arr)

        print("cv loss",cv_loss)
            
        cv_loss_log[epoch] = cv_loss / (shape_cv_data[0] * 2) #Average over number of images and number of classes
        if(cv_loss[0] < prev_loss):
            print("Weights getting better and the loss is ",cv_loss[0],prev_loss)
            prev_loss = cv_loss[0]
            good_wghts = wghts
        else:
            print("Loss is increasing",cv_loss[0])
        
        #time.sleep(3)
    #################      Accuracy Calculation ##################
    
    #Reshape the test data
    #shape_test_data = test_data.shape
    #resize_test_data = np.zeros((shape_test_data[0],size_ftr_vctr))

    #for test_img in range(shape_test_data[0]):
    #    resize_test_data[test_img,:] = test_data[test_img,:].reshape(size_ftr_vctr)

    #test_bias_col = np.ones(shape_test_data[0]).reshape(shape_test_data[0],1)
    #resize_test_data = np.append(test_bias_col,resize_test_data,axis=1)
    #
    pred = 0
    pred_img = np.zeros((shape_test_data[0],1))
    #for test_img in range(shape_test_data[0]):
        #a = test_data[test_img,:] @ good_wghts.T
    a = test_data @ good_wghts.T
    pred_img = np.where(lgst_reg(a) > 0.5, np.ones((shape_test_data[0],1)), np.zeros((shape_test_data[0],1)))
    print(pred_img)
    pred = sum(pred_img == test_label)
        #if(lgst_reg(a) > 0.5):
        #    if(test_label[test_img,0] == 1):
        #        pred_img[
        #        pred += 1
        #else:
        #    if(test_label[test_img,0] == 0):
        #        pred += 1
    
    test_acc = pred / shape_test_data[0]

    plt.plot(np.arange(M),train_loss_log,label='train')
    plt.plot(np.arange(M),cv_loss_log,label='cv')
    plt.legend()
    plt.show()
    
    print("Classification Accuracy is ",test_acc)
    return np.array(cv_loss_log),np.array(train_loss_log), test_acc


def lgst_reg(a):
    '''
    Computes the logistic activation function
    
    '''
    import numpy as np
    p_c0 = 1 / (1 + np.exp(-a))
    #print("LGST",a)
    return p_c0
