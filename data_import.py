import h5py
import numpy as np


def load_data():
    #Loading Data
    train_dataset = h5py.File('data/train_catvnoncat.h5',"r")
    train_x_origin = np.array(train_dataset['train_set_x'][:])
    train_y = np.array(train_dataset['train_set_y'][:])

    test_dataset = h5py.File('data/test_catvnoncat.h5','r')
    test_x_origin = np.array(test_dataset['test_set_x'][:])
    test_y = np.array(test_dataset['test_set_y'][:])

    classes = np.array(test_dataset['list_classes'][:])

    train_y = train_y.reshape((1,train_y.shape[0]))
    test_y = test_y.reshape((1,test_y.shape[0]))

    return train_x_origin, train_y, test_x_origin, test_y, classes

def gettingDimension(train_x_origin, test_x_origin):
    m_train = train_x_origin.shape[0]
    num_px = train_x_origin.shape[1]
    m_test = test_x_origin.shape[0]

    #Printing data
    print("\nNumber of training examples: "+str(m_train))
    print("Number of testing examples: "+str(m_test))
    print("Each image is of size: ( "  + str(num_px) + ", " + str(num_px) + ", 3)")
    print("train_x_origin shape: "+str(train_x_origin.shape))
    print("train_y shape: ( 1, "+ str(m_train) + " )")
    print("test_x_origin shape: "+str(test_x_origin.shape))
    print("test_y shape: ( 1, " + str(m_test) + " )")

    return m_train, num_px, m_test


def flattening_data(train_x_origin, test_x_origin):
    train_x_flatten = train_x_origin.reshape((train_x_origin.shape[0],-1)).T
    test_x_flatten = test_x_origin.reshape((test_x_origin.shape[0],-1)).T

    #standardize the data between 0 and 1
    train_x_flatten = train_x_flatten/255
    test_x_flatten = test_x_flatten/255


    #Printing data
    print("\nData Flattening")
    print("train_x_flatten shape: "+ str(train_x_flatten.shape))
    print("test_x_flatten shape:  "+ str(test_x_flatten.shape))

    return train_x_flatten, test_x_flatten

