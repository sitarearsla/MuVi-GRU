import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from gru import GRU
from muvi_dataset import MuVi_Dataset
import numpy as np
import pandas as pd
from scipy import stats


def validate(model, dataloader):
    """Validate model performance on the validation dataset"""
    pass


def train(model, dataloader):
    """Train model on the training dataset for one epoch"""
    for data in dataloader:
        print(data)


def plot_losses(train_loss, val_loss):
    """Visualize the plots and save them for report."""
    pass


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


def main():
    random_seed = 42
    # Change these paths to the correct paths in your downloaded expert dataset
    data_root = "./"

    save_path_arousal = data_root + "models/audio_model_arousal.ckpt"
    save_path_valence = data_root + "models/audio_model_valence.ckpt"

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
        'batch_size': 256,  # batch size
        'seq_len': 4,  # sequence length
        'dense_units': 256,  # units for dense and lstm blocks
        'lstm_units': 256,
        'lr': 0.0001  # learning rate
    }

    dataset = MuVi_Dataset(data_root, hp['seq_len'], ewe)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * len(dataset)))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    model = GRU()

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=hp['batch_size'], sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=hp['batch_size'], sampler=val_sampler)

    train_losses_valence = []
    train_losses_arousal = []

    val_losses_valence = []
    val_losses_arousal = []

    for i in range(hp['num_epochs']):
        print('Epoch Valence: ' + str(i))
        train_losses_valence.append(train(model, train_loader))
        val_losses_valence.append(validate(model, val_loader))
    torch.save(model, save_path_valence)
    plot_losses(train_losses_valence, val_losses_valence)

    for i in range(hp['num_epochs']):
        print('Epoch Arousal: ' + str(i))
        train_losses_arousal.append(train(model, train_loader))
        val_losses_arousal.append(validate(model, val_loader))
    torch.save(model, save_path_arousal)
    plot_losses(train_losses_arousal, val_losses_arousal)


if __name__ == "__main__":
    main()
