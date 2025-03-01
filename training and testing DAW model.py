 
p = train_x.shape[1]
input_dim = p

BATCH_SIZE = 64
#use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
train_iterator = torch.utils.data.DataLoader(train_x, **params)
valid_iterator = torch.utils.data.DataLoader(val_x, **params)
test_iterator = torch.utils.data.DataLoader(test_x, **params)


#survival function

p = train_x.shape[1] # number of covariates

def weibull_surv(t, a, b , c, d):

    S = np.empty((len(t),len(a)))

    for i in range(len(a)):
        S[:,i] = np.exp(-(a[i] * np.power(t,b[i])) - (c[i] * np.power(t,d[i])))

    return S


def deep_MWD_loss(y, weibull_param_pred):
    epsilon = 1e-10
    time = y[1]  # actual time to event
    status = y[0]  # actual status (censored/dead)
    a = weibull_param_pred[:,0] # alpha
    b = weibull_param_pred[:,1] # beta
    c = weibull_param_pred[:,2] # gamma
    d = weibull_param_pred[:,3] # gamma
    e = (status * torch.log((a * b *(time + epsilon).pow(b - 1)) +
                            (c * d * (time + epsilon).pow(d - 1))))
    return -1 * torch.mean(e)




# Activation    #torch.exp  #F.softplus
def weibull_activate(weibull_param):
    #a = k.exp(weibull_param[:, 0]) # exponential of alpha
    a = torch.exp(weibull_param[:, 0]) # softplus of alpha
    b = torch.exp(weibull_param[:, 1]) # softplus of beta
    c = torch.exp(weibull_param[:, 2]) # softplus of gama
    d = torch.exp(weibull_param[:, 3]) # softplus of gama

    a = torch.reshape(a, (a.shape[0], 1))
    b = torch.reshape(b, (b.shape[0], 1))
    c = torch.reshape(c, (c.shape[0], 1))
    d = torch.reshape(d, (d.shape[0], 1))
    return torch.cat((a, b, c, d), axis=1)


# Autoencoder architecture
lr = 1e-6
input_dim =  p


import torch
import torch.nn as nn

# Define the Autoencoder class (AE)
class AE(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(in_features=p, out_features=35)
        self.encoder_output_layer = nn.Linear(in_features=35, out_features=31)
        self.decoder_hidden_layer = nn.Linear(in_features=31, out_features=35)
        self.decoder_output_layer = nn.Linear(in_features=35, out_features=p)

    def forward(self, features):
        activation = torch.relu(self.encoder_hidden_layer(features))
        code = torch.relu(self.encoder_output_layer(activation))
        activation = torch.relu(self.decoder_hidden_layer(code))
        reconstructed = self.decoder_output_layer(activation)
        return reconstructed


# Define the Mean Squared Error loss for autoencoder
def reconstruction_loss(original, reconstructed):
    return nn.functional.mse_loss(reconstructed, original)

# Assume `train_x` and `val_x` are the training and validation data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move data to the device
train_x = train_x.to(device)
val_x = val_x.to(device)

# Initialize the autoencoder model, optimizer, and learning rate scheduler
model = AE(p=train_x.shape[1]).to(device)
optimizer = torch.optim.RAdam(model.parameters(), lr=0.01)  # Changed to Adam
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)



activation1 = {}
activation2 = {}
activation3 = {}

def get_activation1(name):
    def hook1(model, input, output1):
        activation1[name] = output1.detach()
    return hook1

def get_activation2(name):
    def hook2(model, input, output2):
        activation2[name] = output2.detach()
    return hook2

def get_activation3(name):
    def hook3(model, input, output3):
        activation3[name] = output3.detach()
    return hook3



model1 = AE(p=train_x.shape[1]).to(device)
model2 = AE(p=train_x.shape[1]).to(device)
model3 = AE(p=train_x.shape[1]).to(device)
model1.decoder_output_layer.register_forward_hook(get_activation1('decoder_output_layer'))
model2.decoder_output_layer.register_forward_hook(get_activation2('decoder_output_layer'))
model3.decoder_output_layer.register_forward_hook(get_activation3('decoder_output_layer'))

#x = torch.randn(1, p)
#output = model(x)
#print(activation['fc2'])
output1 = model1(train_x)
output2 = model2(test_x)
output3 = model3(val_x)

recons_train =  activation1['decoder_output_layer']
recons_test =  activation2['decoder_output_layer']
recons_valid =  activation3['decoder_output_layer']


train_x = recons_train
test_x = recons_test
val_x = recons_valid


# Deep additive weibull architecture
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, p)
        self.dropout = nn.Dropout(p=0.2)
        self.layer2 = nn.Linear(p, p)
        self.layer3 = nn.Linear(p, p)  # New hidden layer
        self.layer4 = nn.Linear(p, 4)

    def forward(self, x):
        x = F.tanh(self.layer1(x))
        x = F.tanh(self.layer2(x))
        x = F.tanh(self.layer3(x))  # Apply activation to the new hidden layer
        x = F.softplus(self.layer4(x))  # Or use your custom activation here
        return x



model     = Model(p)
pred_y = model(train_x)

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move data to the device
train_x = train_x.to(device)
train_y = (train_y[0].to(device), train_y[1].to(device))  # Assuming train_y is a tuple (time, status)
val_x = val_x.to(device)
val_y = (val_y[0].to(device), val_y[1].to(device))  # Assuming val_y is a tuple (time, status)

# Initialize model, optimizer, and loss
model = Model(input_dim=train_x.shape[1]).to(device)  # Ensure input_dim is passed correctly
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)  #RAdam
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Training setup
num_epochs = 300
patience = 5
min_delta = 1e-4

loss_train_hist = []
loss_valid_hist = []

best_valid_loss = float('inf')
best_model = None
epoch_since_last_improvement = 0

for epoch in range(num_epochs):
    model.train()
    loss_train = AverageMeter()

    optimizer.zero_grad()
    pred_y = model(train_x)
    loss = deep_MWD_loss(train_y, pred_y)

    # L2 regularization
    l2_lambda = 1e-7  # Adjusted for clarity
    l2_reg = torch.tensor(0., device=device)
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2)

    loss += l2_lambda * l2_reg

    loss.backward()

    # Gradient clipping to prevent gradients from getting too large
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    loss_train.update(loss.item())

    # Validation phase
    model.eval()
    with torch.no_grad():
        loss_valid = AverageMeter()
        pred_y = model(val_x)
        loss = deep_MWD_loss(val_y, pred_y)
        loss_valid.update(loss.item())

    loss_train_hist.append(loss_train.avg)
    loss_valid_hist.append(loss_valid.avg)

    # Adjust the learning rate
    scheduler.step()

    print(f'Epoch {epoch + 1}/{num_epochs}')
    print(f'Train: Loss = {loss_train.avg:.4f}')
    print(f'Valid: Loss = {loss_valid.avg:.4f}')
    print()

    # Check if validation loss has improved
    if loss_valid.avg < best_valid_loss - min_delta:
        best_valid_loss = loss_valid.avg
        best_model = model.state_dict()
        epoch_since_last_improvement = 0
    else:
        epoch_since_last_improvement += 1

    # Early stopping
    if epoch_since_last_improvement >= patience:
        print(f'Validation loss did not improve for {patience} epochs. Stopping early.')
        break

# Optionally, save the best model
if best_model is not None:
    torch.save(best_model, 'best_model.pth')


test_predict = model(test_x) # predict Weibull parameters using covariates
# test_predict = test_predict.resize_(p, 4) # put into (,2) array
test_predict = pd.DataFrame(test_predict.detach().numpy()) # convert to dataframe
test_predict.columns = ["pred_alpha", "pred_beta", "pred_gamma","pred_landa"] # name columns
test_result = df_test.copy()
test_result.reset_index(inplace = True) # reset the index (before concat - probably better way of doing this)
test_result = pd.concat([test_result, test_predict], axis=1) # results = test data plus predictions
test_result.set_index("index", drop=True, inplace=True) # recover the index (after concat - probably better way of doing this)
t_max = df_test["time"].max()
num_vals = max(math.ceil(t_max), 50)
t_vals = np.linspace(0, t_max, num_vals)
surv =  weibull_surv(t_vals, test_result["pred_alpha"].to_numpy(),
                     test_result["pred_beta"].to_numpy(),
                     test_result["pred_gamma"].to_numpy(),
                     test_result["pred_landa"].to_numpy())
surv = pd.DataFrame(data=surv, index=t_vals)

test_time = df_test['time'].values
test_status = df_test['death'].values

#evaluating the model
ev = EvalSurv(surv, test_time, test_status, censor_surv='km')

ev.concordance_td()

ev.integrated_brier_score(time_grid)


