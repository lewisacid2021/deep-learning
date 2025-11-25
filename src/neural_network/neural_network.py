import numpy as np
import matplotlib.pylab as plt
import os

def save_plot(x, y, filename="plot.pdf"):

    save_dir = "./plots"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, filename)

    plt.figure()
    plt.plot(x, y)
    plt.ylim(min(y)-0.1, max(y)+0.1)
    plt.savefig(save_path, format="pdf")
    plt.close()

# 阶跃函数
def step_function(x):
    y=x>0
    return y.astype(np.int32)

# sigmoid函数
def sigmoid(x):
    return 1/(1+np.exp(-x))

# ReLU函数
def relu(x):
    return np.maximum(0, x)

# softmax函数
def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)  # 防止溢出
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a

    return y

x=np.arange(-5.0, 5.0, 0.1)

y=step_function(x)
save_plot(x, y, "step_function.pdf")

y=sigmoid(x)
save_plot(x, y, "sigmoid_function.pdf")

y=relu(x)
save_plot(x, y, "relu_function.pdf")

y=softmax(x)
save_plot(x, y, "softmax_function.pdf")

def init_network():
    network={}
    network['W1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1']=np.array([0.1,0.2,0.3])
    network['W2']=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2']=np.array([0.1,0.2])
    network['W3']=np.array([[0.1,0.3],[0.2,0.4]])
    network['b3']=np.array([0.1,0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)

