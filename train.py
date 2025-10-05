from cProfile import label
from sympy import false, print_glsl
from torch import no_grad
from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

# torch.set_default_device('cuda')


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


"""
##################
### QUESTION 1 ###
##################
"""


def train_perceptron(model, dataset):
    """
    Train the perceptron until convergence.
    You can iterate through DataLoader in order to 
    retrieve all the batches you need to train on.

    Each sample in the dataloader is in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.
    """
    with no_grad():
        while True:
            flag = True
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            "*** YOUR CODE HERE ***"
            for sample in dataloader:
                # print(sample)
                pred = model.get_prediction(sample['x'])
                # print("pred={}, weight_before={}".format(pred, model.w))
                if pred == sample['label']:
                    continue
                
                model.w += sample['x'] * (-pred)
                flag = False
            if flag:
                break
                # print("weight_after={}".format(model.w))

def train_regression(model, dataset):
    """
    Trains the model.

    In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
    batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

    Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.

    Inputs:
        model: Pytorch model to use
        dataset: a PyTorch dataset object containing data to be trained on
        
    """
    "*** YOUR CODE HERE ***"

    # model = torch.nn.Sequential(
    #     torch.nn.Linear(4, 1),
    #     # torch.nn.ReLU()
    #     torch.nn.Flatten(0, 1)
    # )
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    max_loss = 1000

    for i in range(200000):
        max_loss = 0
        for s in dataloader:
            # print(s)
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(s['x'])
            y = s['label']

            # Compute and print loss
            loss = regression_loss(y_pred, y)
            max_loss = max(loss, max_loss)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if i % 100 == 99:
            print(i, max_loss)
        if max_loss <= 0.02:
            break




def train_digitclassifier(model, dataset):
    """
    Trains the model.
    """
    model.train()
    """ YOUR CODE HERE """
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(200000):
        for s in dataloader:
            # print(s)
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(s['x'])
            y = s['label']

            # Compute and print loss
            loss = digitclassifier_loss(y_pred, y)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        accuary = dataset.get_validation_accuracy()
        if i % 99:
            print("i={}, accuarcy={}".format(i, accuary))
        if accuary >= 0.98:
            break    


def train_languageid(model, dataset):
    """
    Trains the model.

    Note that when you iterate through dataloader, each batch will returned as its own vector in the form
    (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
    get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
    that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
    as follows:

    movedim(input_vector, initial_dimension_position, final_dimension_position)

    For more information, look at the pytorch documentation of torch.movedim()
    """
    model.train()
    "*** YOUR CODE HERE ***"
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 创建调度器 - 监控验证准确率
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max',      # 准确率越大越好
        patience=5,      # 5轮无改善就衰减
        factor=0.5,      # 学习率减半
        min_lr=1e-5
    )

    for i in range(200):
        m_loss = 0
        for s in dataloader:
            # print("s={}, x={}, y={}".format(s, s['x'].shape, s['label'].shape))

            # print(s)
            # Forward pass: Compute predicted y by passing x to the model

            xs = movedim(s['x'], 1, 0)
            y_pred = model(xs)
            y = s['label']
            # print("i={}, y_pred={}, y={}".format(i, y_pred, y))
            # print(f"y_pred shape: {y_pred.shape}")  # Debug 
            # print(f"y shape: {y.shape}")            # Debug

            # Compute and print loss
            loss = languageid_loss(y_pred, y)
            m_loss = max(loss, m_loss)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        accuary = dataset.get_validation_accuracy()
        
        scheduler.step(accuary)  # 传入监控的指标值
        current_lr = optimizer.param_groups[0]['lr']
        if i % 99:
            print("i={}, accuray={}, loss={}, lr={}".format(i, accuary, m_loss, current_lr))

        if accuary >= 0.81:
            break


def Train_DigitConvolution(model, dataset):
    """
    Trains the model.
    """
    """ YOUR CODE HERE """
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(50):
        m_loss = 0
        for s in dataloader:
            # print(s)
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(s['x'])
            y = s['label']

            # Compute and print loss
            loss = digitconvolution_Loss(y_pred, y)
            m_loss = max(m_loss, loss)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        accuary = dataset.get_validation_accuracy()
        if i % 99:
            print("i={}, accuarcy={}, loss={}".format(i, accuary, loss))
        if accuary >= 0.98:
            break    