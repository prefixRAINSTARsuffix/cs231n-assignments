import numpy as np

def load_CIFAR10(file):
    import pickle
    train_data = None
    train_label = None
    
    for i in range(1,6):
        with open(file+'data_batch_{}'.format(i),'rb') as fo:
            dict = pickle.load(fo,encoding='latin1')
        if(train_data is None):
            train_data = dict['data']
            train_label = dict['labels']
        else:
            train_data = np.concatenate((train_data,dict['data']),axis=0)
            train_label = np.concatenate((train_label,dict['labels']),axis=0)
    
    with open(file+'test_batch','rb') as fo:
        dict = pickle.load(fo,encoding='latin1')
    
    return train_data.astype("float"),np.array(train_label).astype("float"),dict['data'].astype("float"),np.array(dict['labels']).astype("float")