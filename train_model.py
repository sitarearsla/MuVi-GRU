import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from gru import Audio_GRU
from muvi_dataset import MuVi_Dataset
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from audtorch.metrics.functional import concordance_cc


def validate(model, dataloader, criterion, valence=False):
    """Validate model performance on the validation dataset"""
    model.eval()
    total_loss = 0.0
    # print("dataset len: ", len(dataloader.dataset))
    with torch.no_grad():
        for batch in dataloader:
            x_data, y_arousal, y_valence = batch
            if torch.cuda.is_available():
                if valence:
                    x_data, y_valence = x_data.cuda(), y_valence.cuda()
                else:
                    x_data, y_arousal = x_data.cuda(), y_arousal.cuda()
            pred = model(x_data)
            y = y_valence if valence else y_arousal
            loss = criterion(pred, y)
            total_loss += loss
            ccc = concordance_cc(y.clone().detach().reshape(-1), pred.clone().detach().reshape(-1))

        avg_total_loss = total_loss / len(dataloader.dataset)
        print('Validation - CCC: ', ccc.item())
        print('Validation - MSE:', avg_total_loss.item())
    return avg_total_loss.item(), ccc.item()


def train(model, dataloader, optimizer, criterion, valence=False):
    """Train model on the training dataset for one epoch"""

    model.train()
    total_loss = 0.0
    # print("dataset len: ", len(dataloader.dataset))
    for batch_id, batch in enumerate(dataloader):
        optimizer.zero_grad()
        x_data, y_arousal, y_valence = batch

        if torch.cuda.is_available():
            if valence:
                x_data, y_valence = x_data.cuda(), y_valence.cuda()
            else:
                x_data, y_arousal = x_data.cuda(), y_arousal.cuda()
        pred = model(x_data)
        y = y_valence if valence else y_arousal
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        # Show progress
        # if batch_id % 5 == 0 or batch_id == len(dataloader):
            # print('[{}/{}] loss: {:.8}'.format(batch_id, len(dataloader), loss.item()))
        ccc = concordance_cc(y.clone().detach().reshape(-1), pred.clone().detach().reshape(-1))
        total_loss += loss

    avg_total_loss = total_loss / len(dataloader.dataset)
    print('Train - CCC: ', ccc.item())
    print('Train - MSE:', avg_total_loss.item())

    return avg_total_loss.item(), ccc.item()


def plot_losses(train_loss, val_loss):
    """Visualize the plots and save them for report."""

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('GRU Loss Plots')
    fig.supxlabel('Epoch')
    fig.supylabel('Loss')
    ax1.set_title('MSE Loss')
    ax2.set_title('Concordance Correlation Coefficient')

    train_losses_mse = train_loss['mse']
    train_losses_ccc = train_loss['ccc']

    val_losses_mse = val_loss['mse']
    val_losses_ccc = val_loss['ccc']

    train_losses_mse_x = range(len(train_losses_mse))
    train_losses_ccc_x = range(len(train_losses_ccc))

    val_losses_mse_x = range(len(val_losses_mse))
    val_losses_ccc_x = range(len(val_losses_ccc))

    ax1.plot(train_losses_mse_x, train_losses_mse, 'b', label='train')
    ax1.plot(val_losses_mse_x, val_losses_mse, 'r', label='validation')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    ax1.set_facecolor("white")

    ax2.plot(train_losses_ccc_x, train_losses_ccc, 'b', label='train')
    ax2.plot(val_losses_ccc_x, val_losses_ccc, 'r', label='validation')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    ax2.set_facecolor("white")

    fig.tight_layout()
    plt.savefig('gru_loss_single_modal.png')

def getEWE(data):
    """
    taken from the original implementation
    calculate Evaluator Weighted Estimator as detailed in prof Desmond's paper.
    """

    ratingsArray_arousal, ratingsArray_valence = [], []

    for n, g in data.groupby('participant_id'):
        ratingsArray_arousal.append(g.iloc[0:118].arousal)
        ratingsArray_valence.append(g.iloc[0:118].valence)

    averagedRating_arousal = np.array(ratingsArray_arousal).mean(axis=0)
    std_arousal = np.array(ratingsArray_arousal).std(axis=0)

    averagedRating_valence = np.array(ratingsArray_valence).mean(axis=0)
    std_valence = np.array(ratingsArray_valence).std(axis=0)

    # weight each individual's ratings
    weightedRatings_arousal, weightedRatings_valence = [], []
    weights_arousal, weights_valence = [], []

    for n, g in data.groupby('participant_id'):
        weight_arousal, _ = stats.pearsonr(g.iloc[0:118].arousal, averagedRating_arousal)

        # there's a slightly worrisome no. of ratings with corr < 0 (arousal: 136, valence: 172), but I don't see what can be done...
        if weight_arousal > 0:
            weightedRatings_arousal.append(g.iloc[0:118].arousal * weight_arousal)
            weights_arousal.append(weight_arousal)

        weight_valence, _ = stats.pearsonr(g.iloc[0:118].valence, averagedRating_valence)

        # if not np.isnan(weight_valence):
        if weight_valence > 0:
            weightedRatings_valence.append(g.iloc[0:118].valence * weight_valence)
            weights_valence.append(weight_valence)

    ewe_arousal = np.array(weightedRatings_arousal).sum(axis=0) * (1 / sum(weights_arousal))
    ewe_valence = np.array(weightedRatings_valence).sum(axis=0) * (1 / sum(weights_valence))

    return ewe_arousal, ewe_valence, std_arousal, std_valence

def save_results(train_loss, val_loss):
    # creating a Dataframe object
    df_train = pd.DataFrame(train_loss)
    df_validation = pd.DataFrame(val_loss)
    df_train.to_csv('train_loss.csv')
    df_validation.to_csv('validation_loss.csv')

def main():
    random_seed = 42
    # Change these paths to the correct paths in your downloaded expert dataset
    data_root = "./"

    save_path = data_root + "models/audio_model_valence.ckpt"

    # av_data includes the dynamic (continuous) annotations for Valence and Arousal.
    av_df = pd.read_csv(data_root + "av_data.csv")
    # keep only music type
    av_df = av_df[av_df['media_modality'] == 'music']
    # drop group in av_df with length==111
    drop_indexes = av_df[(av_df.participant_id == 23) & (av_df.media_id == 'U0CGsw6h60k')].index
    av_df.drop(drop_indexes, inplace=True)
    # create rowIDs for av_df
    av_df['rowID'] = list(zip(av_df["media_id"], av_df["media_modality"], av_df["participant_id"]))
    # drop low-quality annotations (cursor not moved at all)
    constant_inputs = []

    for name, group in av_df.groupby(['media_id', 'media_modality', 'participant_id']):
        p, _ = stats.pearsonr(group.iloc[0:118].arousal, group.iloc[0:118].valence)

        if np.isnan(p):
            constant_inputs.append(name)

    av_df = av_df[~av_df.rowID.isin(constant_inputs)]

    # calculate overall EWE ratings for all media items
    ewe = []

    for name, group in av_df.groupby(['media_id', 'media_modality']):
        arousal, valence, arousal_std, valence_std = getEWE(group)
        ewe.append([name[0], name[1], arousal, valence, arousal_std, valence_std])

    # hyper-parameters
    hp = {
        'num_epochs': 100,  # number of epochs
        'batch_size': 8,  # batch size
        'seq_len': 4,  # sequence length
        'num_layers' : 4, # stacking 4 RNNs
        'lr': 1e-5, # learning rate
        'dropout': 0.5,
        'embedding_size': 256,
        'hidden_size': 256,
        'output_size' : 114,
        'input_size': 988,
        'bidirectional': False,
    }

    print('Preparing the dataloaders...')
    model = Audio_GRU(hp)
    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=hp['lr'])
    criterion = nn.MSELoss()
    dataset = MuVi_Dataset(data_root, hp['seq_len'], ewe)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * len(dataset)))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=hp['batch_size'], sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(dataset, batch_size=hp['batch_size'], sampler=val_sampler)


    train_losses_mse = []
    val_losses_mse = []
    train_losses_ccc = []
    val_losses_ccc = []

    print('Starting GRU training ...')
    # Early stopping
    last_loss = 100
    patience = 7
    trigger_times = 0

    for i in range(hp['num_epochs']):
        print('-------------------------------------------------------')
        print('Epoch: ' + str(i))
        total_loss, ccc_train = train(model, train_loader, optimizer, criterion)
        train_losses_mse.append(total_loss)
        train_losses_ccc.append(ccc_train)

        # Early stopping
        current_loss, ccc_val = validate(model, val_loader, criterion)
        val_losses_mse.append(total_loss)
        val_losses_ccc.append(ccc_val)

        if current_loss > last_loss:
            trigger_times += 1
            print('Trigger Times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!\n Saving the model.')
                return

        else:
            print('trigger times: 0')
            trigger_times = 0

        last_loss = current_loss

    train_loss = {
        'mse': train_losses_mse,
        'ccc': train_losses_ccc,
    }

    val_loss = {
        'mse': val_losses_mse,
        'ccc': val_losses_ccc,
    }

    print("Finished training. . . ")
    torch.save(model, save_path)
    save_results(train_loss, val_loss)
    plot_losses(train_loss, val_loss)


if __name__ == "__main__":
    main()
