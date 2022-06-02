from typing import Tuple, List
from unicodedata import bidirectional
from transformers import  AutoModel
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_dataset import CustomDataset
from torch.utils.data import Dataset, DataLoader
from data_reader import read_dataset_convabuse, get_dataset_bert, read_dataset_tcc, read_dataset_alyt
from sklearn.metrics import accuracy_score, f1_score

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()
        self.hate_bert = AutoModel.from_pretrained("GroNLP/hateBERT")

        self.fc1 = nn.Linear(num_inputs, 512)
        self.drop1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 128)
        self.drop2 = nn.Dropout(0.25)
        self.fully_connected = nn.Linear(128, num_outputs)

    def forward(self, x):
        
        x = self.hate_bert(x)[0]
        x = th.flatten(x, 1) # flatten all dimensions except batch
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = self.fully_connected(x)
        return x

class Net2(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()
        self.hate_bert = AutoModel.from_pretrained("GroNLP/hateBERT")

        self.lstm1 = th.nn.LSTM(num_inputs, num_inputs // 2, bidirectional = True, num_layers = 2, dropout = 0.30)
        self.drop1 = th.nn.Dropout(0.30)

        self.att_layer = th.nn.Linear(num_inputs, num_inputs)
        self.fc = th.nn.Linear(num_inputs, 100)
        self.drop2 = th.nn.Dropout(0.30)

        self.final_layer = th.nn.Linear(100, num_outputs)

    def forward(self, x):
        
        x = self.hate_bert(x)[0]

        x, (_, _) = self.lstm1(x)
        x = F.dropout(x, 0.25)

        attention = self.att_layer(x)
        attention = th.exp(attention)
        attention = attention / (th.sum(attention, dim=1, keepdim =True) + 1e-5).type(th.float32)
        x = x * attention
        x = th.sum(x, dim=1)

        x = self.fc(x)
        x = F.dropout(x, 0.25)
        x = self.final_layer(x)

        return x

def train_loop(train_set, valid_set, inputs: int = 768, outputs: int = 2, epochs: int = 10, model_variation: int = 2, model_save_path: str = './models/trained_model.pt', pre_treined_model_save_path = None):

    net = None
    if model_variation == 2:
        net = Net2(inputs, outputs)
    else:
        net = Net(inputs, outputs)

    if pre_treined_model_save_path is not None:
        net.load_state_dict(th.load(pre_treined_model_save_path))


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = 1e-5)

    dataset = CustomDataset(train_set[0], train_set[1])
    valid_set = CustomDataset(valid_set[0], valid_set[1])
    dataloader = DataLoader(dataset, batch_size=32,
                        shuffle=True, num_workers=0)
    validloader = DataLoader(valid_set, batch_size=4,
                        shuffle=False, num_workers=0)

    net.to(device)
    best_val_loss = 1.0

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            x = data['x'].to(device)
            y = data['y'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i + 1):.3f}')
            # running_loss = 0.0
        
        running_loss = 0.0
        cnt = 0

        for i, data in enumerate(validloader, 0):
            
            x = data['x'].to(device)
            y = data['y'].to(device)

            optimizer.zero_grad()

            outputs = net(x)

            running_loss += loss.item()
            cnt = cnt + 1
        
        print("EPOCH: ", epoch + 1, "VALID LOSS: ", running_loss / cnt)

        if running_loss / cnt < best_val_loss:

            th.save(net.state_dict(), model_save_path)


def score(test_set: List[Tuple[str, int]], inputs: int = 768, outputs: int = 2, model_variation: int = 2, model_save_path: str = './models/trained_model.pt'):

    if model_variation == 2:
        net = Net2(inputs, outputs)
    else:
        net = Net(inputs, outputs)

    net.load_state_dict(th.load(model_save_path))

    net.to(device)
    x_test, y_test, max_len = get_dataset_bert(test_set, outputs)

    y_real = []
    y_pred = []

    for i, x in enumerate(x_test):
        x = th.tensor([x]).to(device)

        y_out = net.forward(x).cpu().detach().numpy()

        y_out = np.argmax(y_out)

        y_real.append(np.argmax(y_test[i]))
        y_pred.append(y_out)
    
    print("accuracy", accuracy_score(y_real, y_pred))
    print("macro_F1", f1_score(y_real, y_pred, average='macro'))

def run_second_iteration_experiments(version: int = 1):

    percentage = 0.2
    max_len = 100
    inputs = 768

    if version == 1:
        inputs = inputs * max_len

    train_set, validation_set, test_set = read_dataset_convabuse('./datasets/ConvAbuseEMNLPtrain.csv', './datasets/ConvAbuseEMNLPvalid.csv', './datasets/ConvAbuseEMNLPtest.csv')
    
    x, y, max_len = get_dataset_bert(train_set, max_len=max_len)
    x_valid, y_valid, max_len = get_dataset_bert(validation_set, max_len=max_len)

    train_loop(
        (x, y),
        (x_valid, y_valid),
        inputs=inputs,
        outputs=2, epochs=10,
        model_variation=version,
        model_save_path='./models_torch/convabuse_version' + str(version) + "_hatebert.pt"
    )

    print("~~~~~ ConvAbuse Scores: ~~~~")
    score(
        test_set, 
        inputs=inputs,
        outputs=2,
        model_variation=version,
        model_save_path='./models_torch/convabuse_version' + str(version) + "_hatebert.pt"
    )

    train_set, test_set = read_dataset_tcc('./datasets/Dataset40CombinedV2.csv')

    x, y, max_len = get_dataset_bert(train_set, max_len=max_len)

    train_loop(
        (x[:int(len(train_set) * (1 - percentage))], y[:int(len(train_set) * (1 - percentage))]),
        (x[int(len(train_set) * (1 - percentage)):], y[int(len(train_set) * (1 - percentage)):]),
        inputs=inputs, outputs=2, epochs=10, model_variation=version, model_save_path='./models_torch/tcc_version' + str(version) + '_hatebert.pt'
    )
    print("~~~~~ TCC Scores: ~~~~")
    score(
        test_set, 
        inputs=inputs,
        outputs=2,
        model_variation=version,
        model_save_path='./models_torch/tcc_version' + str(version) + '_hatebert.pt'
    )

    
    train_set, test_set = read_dataset_alyt('./datasets/ALYT_data.csv')
    x, y, max_len = get_dataset_bert(train_set, target=3)


    train_loop(
        (x[:int(len(train_set) * (1 - percentage))], y[:int(len(train_set) * (1 - percentage))]),
        (x[int(len(train_set) * (1 - percentage)):], y[int(len(train_set) * (1 - percentage)):]),
        inputs=inputs, outputs=3, epochs=10, model_variation=version, model_save_path='./models_torch/alyt_version' + str(version) + '_hatebert.pt'
    )
    print("~~~~~ ALYT Scores: ~~~~")
    score(test_set, inputs=inputs, outputs=3, model_variation=version, model_save_path='./models_torch/alyt_version' + str(version) + '_hatebert.pt')

def run_transfer_learning_experiment(version: int = 1, pre_trained_model_path: str = './models_torch/tcc_version1_hatebert.pt'):

    max_len = 100
    inputs = 768

    if version == 1:
        inputs = inputs * max_len

    train_set, validation_set, test_set = read_dataset_convabuse('./datasets/ConvAbuseEMNLPtrain.csv', './datasets/ConvAbuseEMNLPvalid.csv', './datasets/ConvAbuseEMNLPtest.csv')
    
    x, y, max_len = get_dataset_bert(train_set, max_len=max_len)
    x_valid, y_valid, max_len = get_dataset_bert(validation_set, max_len=max_len)

    train_loop(
        (x, y),
        (x_valid, y_valid),
        inputs=inputs,
        outputs=2, epochs=10,
        model_variation=version,
        model_save_path='./models_torch/convabuse_version' + str(version) + "from_tcc_hatebert.pt",
        pre_treined_model_save_path=pre_trained_model_path
    )
    score(
        test_set, 
        inputs=inputs,
        outputs=2,
        model_variation=version,
        model_save_path='./models_torch/convabuse_version' + str(version) + "from_tcc_hatebert.pt"
    )

    train_set, test_set = read_dataset_tcc('./datasets/Dataset40CombinedV2.csv')

    score(
        test_set, 
        inputs=inputs,
        outputs=2,
        model_variation=version,
        model_save_path='./models_torch/convabuse_version' + str(version) + "from_tcc_hatebert.pt"
    )

if __name__ == "__main__":

    run_second_iteration_experiments()









