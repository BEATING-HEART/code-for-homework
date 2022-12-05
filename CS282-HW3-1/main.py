import pandas as pd
import numpy as np
# from sklearn import linear_model
import matplotlib.pyplot as plt

def get_loss(X, y, theta):
    m = len(y)
    # return (1/(2*m)) * np.square(np.linalg.norm(np.dot(train_x, theta) - train_y))
    return np.square(np.linalg.norm(np.dot(X, theta) - y))


if __name__ == "__main__":
    data = pd.read_csv('data/data_set.csv')
    # print(data)
    TRAIN_NUM = 200

    train_x = np.insert(data.iloc[:TRAIN_NUM,:4].to_numpy(), 0, np.ones(TRAIN_NUM), axis=1)
    train_y = data.iloc[:TRAIN_NUM,4:].to_numpy()

    test_x = np.insert(data.iloc[TRAIN_NUM:, :4].to_numpy(), 0, np.ones(249-TRAIN_NUM), axis=1)
    test_y = data.iloc[TRAIN_NUM:, 4:].to_numpy()
    
    theta = np.random.randn(5).reshape(5,1)
    # print(theta)
    
    # cost = np.square(np.linalg.norm(np.dot(train_x, theta) - train_y))
    
    for i in range(5000000):
        # theta -= 0.1 * np.ones(5).reshape(5,1)
        gradient = (train_x.T).dot(train_x.dot(theta)-train_y)
        # print(gradient)
        theta =  theta - 0.000000001 * gradient
        if((i+1) % 100 == 0):
            print(i+1)
            # print(theta)
            # print(train_x.dot(theta)-train_y)
            print(get_loss(train_x, train_y, theta))
            print('*******')
    
    pred_label = train_x.dot(theta)
    
    test_pred = test_x.dot(theta)
    test_pred_mean = np.mean(test_pred)
    # print(test_pred_mean)

    # test_pred_mean_arr = np.full_like(test_pred, test_pred_mean)

    _var = np.square(np.linalg.norm(test_pred - test_pred_mean_arr))

    bias = np.square(np.linalg.norm(test_pred_mean_arr - test_y.flatten()))
    # print(test_pred_mean_arr)
    # print(test_y.flatten())
    print(_var)
    print(bias)      
    # print(pred_label)
    # print(train_y)
    # print(get_loss(train_x, train_y, theta))