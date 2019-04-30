import numpy as np
import matplotlib.pyplot as plt

def balanced_sample_maker(X, y, sample_size, random_seed=42):
    uniq_levels = np.unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_levels}

    if not random_seed is None:
        np.random.seed(random_seed)

    # find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx
    # oversampling on observations of each label
    balanced_copy_idx = []
    for gb_level, gb_idx in groupby_levels.items():
        over_sample_idx = np.random.choice(gb_idx, size=sample_size, replace=True).tolist()
        balanced_copy_idx+=over_sample_idx
    np.random.shuffle(balanced_copy_idx)
    data_train=X[balanced_copy_idx]
    labels_train=y[balanced_copy_idx]
    if  ((len(data_train)) == (sample_size*len(uniq_levels))):
        print('number of sampled example ', sample_size*len(uniq_levels), 'number of sample per class ', sample_size, ' #classes: ', len(list(set(uniq_levels))))
    else:
        print('number of samples is wrong ')
    print('data train',data_train.shape, labels_train)	
    return (data_train,labels_train,balanced_copy_idx)

def next_picker(nX,ny,old_train,old_label,incorrect,prob):
    x=nX[incorrect]
    y=ny[incorrect]
    uniq_levels=np.unique(y)
    uniq_counts={lev:sum(y==lev) for lev in uniq_levels}
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx
    # oversampling on observations of each label
    balanced_copy_idx = []
    for gb_level, gb_idx in groupby_levels.items():
        #print(type(gb_idx),len(gb_idx),gb_idx)
        #print([k for k in gb_idx])
        #print(prob.keys())
        prob_lev={k:prob[k] for k in gb_idx}
        
        maxx=max(prob_lev.values())
        keys = [x for x,y in prob_lev.items() if y ==maxx]
        if len(keys)>1:
            keys=[(np.random.choice(np.array(keys)))]
            
        balanced_copy_idx+=keys
    print('y',y)
    print('balanced copy',balanced_copy_idx)
    data_train=x[balanced_copy_idx]
    labels_train=np.reshape(y[balanced_copy_idx],(10,1))
    newX =nX[np.setdiff1d(np.arange(nX.shape[0]), balanced_copy_idx)]
    newy = ny[np.setdiff1d(np.arange(ny.shape[0]), balanced_copy_idx)]
    print('old train',old_train.shape,'data train',data_train.shape)
    print('old label',old_label.shape,'labels train',labels_train.shape)
    data_train=np.concatenate((old_train,data_train),axis=0)
    labels_train=np.concatenate((old_label,labels_train),axis=0)
    return (data_train,labels_train,newX,newy)

 
