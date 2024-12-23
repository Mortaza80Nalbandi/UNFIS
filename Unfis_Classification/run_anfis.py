from anfis import ANFIS
import anfis_generator,plottingtools
import torch
from helpers import  _FastTensorDataLoader
import torch.nn as nn
import torch.nn.functional as F
from time import time
from sklearn.metrics import classification_report
def fit(model,train_data,valid_data,epochs,batch_size: int = 2,shuffle_batches: bool = True,interval=50):
    device = model.device
    optimizer = torch.optim.Adam(params=model.parameters(),lr=0.01)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    # get dataloader
    train_dl = _FastTensorDataLoader(train_data,batch_size, shuffle_batches)
    valid_dl = _FastTensorDataLoader(valid_data,batch_size, shuffle_batches)
    print(f"Train anfis on {len(train_dl.dataset[0])} samples, validate on {len(valid_dl.dataset[0])} samples")
    valid_loss = []
    train_loss = []
    # main training loop
    now = time()
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss_epoch = 0
        for  xb_train, yb_train in train_dl:
            # send tensors to device (cpu/gpu)
            xb_train = xb_train.to(device)
            yb_train = yb_train.to(device)
            # forward pass & loss calculation
            optimizer.zero_grad()
            train_pred,reconstruct = model( xb_train)
            loss1 = criterion1(train_pred, yb_train)
            loss2 = criterion2(reconstruct,xb_train)
            loss = loss1+loss2
            # perform backward, update weights, zero gradients
            loss.backward()
            optimizer.step()
            train_loss_epoch+=loss.detach().cpu()
        train_loss.append(train_loss_epoch)
        # Validation
        valid_loss_epoch = 0
        with torch.no_grad():
            model.eval()
            xb_valid, yb_valid = valid_dl.dataset
            xb_valid = xb_valid.to(device)
            yb_valid = yb_valid.to(device)
            y_pred_valid,reconstruct_valid = model( xb_valid)
            # TODO: should not be a list.
            valid_loss_epoch +=  (criterion1(y_pred_valid,yb_valid).detach().cpu()+criterion2(reconstruct_valid,xb_valid).detach().cpu())
        valid_loss.append(valid_loss_epoch)
        if(epoch %interval ==0 ):
            new_Time = round(time() - now, 1)
            now = time()
            print("Episode ",epoch)
            print("train loss : ",train_loss_epoch)
            print("valid loss : ", valid_loss_epoch)
            if (epoch != 0):
                print("Time :  ",(new_Time/10))
            print("*******************")

    return [train_loss,valid_loss]
def predict(model, input):
    model.pred = "pred"
    # get dataloader
    dataloader = _FastTensorDataLoader(input, batch_size=1000)
    model.eval()
    # predict
    with torch.no_grad():
        model.eval()
        xb = dataloader.dataset
        xb = xb.to(model.device)
        y_pred_,_ = model(xb)
        _, predicted = torch.max(y_pred_.cpu().data, 1)
    return predicted

if __name__ == "__main__":
    data_ids = ['iris', 'wine']
    dataid = data_ids[0]
    # plain Vanilla ANFIS
    MEMBFUNCS, n_input = anfis_generator.get_membsFuncs(dataid)
    # generate some data (mackey chaotic time series)
    X, X_train, X_valid, y, y_train, y_valid = anfis_generator.gen_data(dataid)
    # create model and fit model
    model = ANFIS(membfuncs=MEMBFUNCS, n_input=n_input,classes=3)
    losses = fit(model,train_data=[X_train, y_train], valid_data=[X_valid, y_valid], epochs=100,interval = 10)
    # predict data
    y_pred = predict(model,X)
    print(y_pred)
    print(y)
    # plot and show Results
    plottingtools.plt_learningcurves(losses, save_path='img/learning_curves_'+dataid+'.png')
    print(classification_report(y, y_pred,labels=[0,1,2]))
