import matplotlib.pyplot as plt
import matplotlib.animation as animate
import pandas as pd

'''
Building a simple classifier on whether a bug is a ladybird/caterpillar depending
on the width of the bug
'''

# x = feature to change
# E = error = (desired_value - initial_value)
# L = learning rate (to minimise overshooting)
# returns the number to change the gradient by
def change_A(L, E, x):
    return L*(E / x)

# t = target_value
# y = current_value
# returns the error between the desired line and the current line
def error(t, y):
    return t - y

# t = target_value
# x = feature to train
# y = label to train
# returns the gradient to change the current classifying line by
def train(t, x, y):
    E = error(t, y)
    L = 0.5

    y_new = y + change_A(L, E, x)
    return y_new

# x0 = initial_X
# y0 = initial_Y
# new coefficient for the line
def line(x0, y0):
    # new classifier line
    # y = Ax, A_new = y/x, y_new = (A_new)x 
    A_new = y0/x0
    return A_new

if __name__ == '__main__':
    # dataset
    dataset = {'bug': ['ladybird', 'caterpillar'],
                'width': [3, 1],
                'length': [1, 3]}

    df = pd.DataFrame(dataset)
    print(df)
    
    # classification line => y = AX
    # pick a random value for A
    df = df.drop(['bug'], axis=1)
    X = df.values[0]
    A = 0.25

    # initial guess line
    y = A*X

    # training to get new y's
    ladybird_train = train(1.1, X[0], y[0])
    caterpillar_train = train(2.9, X[1], y[1])

    # setting the coefficient for the new lines
    A_ladybird = line(X[0], ladybird_train)
    A_caterpillar = line(X[1], caterpillar_train)
    print('[Classifier lines]')
    print('y1 = {}x'.format(A_ladybird*X[0]))
    print('y2 = {}x'.format(A_caterpillar*X[1]))

    # plotting the classification lines
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot([0, X[0], X[0]+1], [0, A_ladybird*X[0], A_ladybird*(X[0]+1)], color='red', label='Ladybird')
    ax1.plot([0, X[1], X[1]+1], [0, A_caterpillar*X[1], A_caterpillar*(X[1]+1)], color='green', label='Caterpillar')
    ax1.legend(loc='best')

    # plotting the dataset
    plt.plot(df['width'][0], df['length'][0], color='red', marker='o', linestyle='none')
    plt.plot(df['width'][1], df['length'][1], color='green', marker='o', linestyle='none')
    
    plt.xlabel('width')
    plt.ylabel('length')

    
    plt.show()

