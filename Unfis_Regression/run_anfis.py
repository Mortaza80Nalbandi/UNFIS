from anfis import ANFIS
import anfis_generator,plottingtools
import torch
from helpers import  _FastTensorDataLoader
import torch.nn.functional as F
from sklearn.metrics import classification_report
def fit(model,train_data,valid_data,epochs,batch_size: int = 16,shuffle_batches: bool = True,interval=50,  disable_output: bool = False):
    if len(train_data) == 2:
        train_data = [train_data[0], train_data[0], train_data[1]]
        valid_data = [valid_data[0], valid_data[0], valid_data[1]]
    device = model.device
    optimizer = torch.optim.Adam(params=model.parameters())
    loss_functions = torch.nn.MSELoss(reduction='mean')
    # get dataloader
    train_dl = _FastTensorDataLoader(train_data,batch_size, shuffle_batches)
    valid_dl = _FastTensorDataLoader(valid_data,batch_size, shuffle_batches)


    # print
    if not disable_output:
        print(
            f"Train anfis on {len(train_dl.dataset[0])} samples, validate on {len(valid_dl.dataset[0])} samples")
    valid_loss = []
    train_loss = []
    # main training loop (via tqdm progress bar)
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss_epoch = 0
        for sb_train, xb_train, yb_train in train_dl:
            # send tensors to device (cpu/gpu)
            sb_train = sb_train.to(device)
            xb_train = xb_train.to(device)
            yb_train = yb_train.to(device)
                # forward pass & loss calculation
            train_pred = model(sb_train, xb_train)
            loss = loss_functions(train_pred, yb_train)

            # perform backward, update weights, zero gradients
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss_epoch+=loss.detach().cpu()
        train_loss.append(train_loss_epoch)
        # Validation
        valid_loss_epoch = 0
        with torch.no_grad():
            model.eval()
            sb_valid, xb_valid, yb_valid = valid_dl.dataset
            sb_valid = sb_valid.to(device)
            xb_valid = xb_valid.to(device)
            yb_valid = yb_valid.to(device)
            y_pred_valid = model(sb_valid, xb_valid)
            # TODO: should not be a list.
            valid_loss_epoch += loss_functions(yb_valid, y_pred_valid).detach().cpu()
        valid_loss.append(valid_loss_epoch)
        if(epoch %interval ==0 ):
            print("Episode ",epoch)
            print("train loss : ",train_loss_epoch)
            print("valid loss : ", valid_loss_epoch)
            print("*******************")

    return [train_loss,valid_loss]
def predict( model,input):
    if type(input) == torch.Tensor:
        input = [input, input]
    elif type(input) == list and len(input) == 1:
        input = [input[0], input[0]]
    elif type(input) == list and len(input) == 2:
        pass
    else:
        raise ValueError(f'input must be either a torch tensor (for ANFIS), a list of 1 torch tensor (for ANFIS) or a list of two torch tensors (for S-ANFIS).')

    # get dataloader
    dataloader = _FastTensorDataLoader(input, batch_size=1000)
    model.eval()
    # predict
    with torch.no_grad():
        model.eval()
        sb, xb = dataloader.dataset
        sb = sb.to(model.device)
        xb = xb.to(model.device)
        y_pred_ = model(sb, xb)
    return y_pred_.cpu()

if __name__ == "__main__":
    data_ids = ['mackey', 'sinc', 'sin']
    dataid = data_ids[2]
    # plain Vanilla ANFIS
    MEMBFUNCS, n_input = anfis_generator.get_membsFuncs(data_id=dataid)
    # generate some data (mackey chaotic time series)
    X, X_train, X_valid, y, y_train, y_valid = anfis_generator.gen_data(data_id=dataid, n_obs=300, n_input=n_input)
    # create model and fit model
    model = ANFIS(membfuncs=MEMBFUNCS, n_input=n_input, scale='Std',classes=3)
    losses = fit(model,train_data=[X_train, y_train], valid_data=[X_valid, y_valid], epochs=200,interval = 25)
    # predict data
    y_pred = predict(model,X)

    # plot Results
    plottingtools.plt_learningcurves(losses, save_path='img/learning_curves_' + dataid + '.png')
    plottingtools.plt_prediction(y, y_pred, save_path='img/prediction_'+dataid+'.png')
