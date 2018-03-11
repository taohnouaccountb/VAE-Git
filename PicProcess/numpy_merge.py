import numpy as np
def random_split_data(data, proportion):
    size = data.shape[0]
    s = np.random.permutation(size)
    split_idx = int(proportion * size)
    return data[s[:split_idx]], data[s[split_idx:]]

for i in range(1,11):
    filename='train_data_batch_'+str(i)
    temp=np.load(filename)
    if i==1:
        data=temp['data']
    else:
        data=np.vstack((data,temp['data']))
data=np.reshape(data,(-1,32,32,3))
train, test = random_split_data(data, 0.8)
np.save('train.npy',train)
np.save('test.npy',test)