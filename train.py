# Train Clients
import warnings

warnings.simplefilter('ignore', UserWarning)

# Common
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Union
from pathlib import Path
import copy
import shutil
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
from torchsummary import summary
from sklearn import metrics
import torch.nn.functional as F

# Torch
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

# Utils
from utils import read_py_config

# Loss
from loss import RegisterLoss, mk_loss

# Optimizer
from optim import RegisterOptim, mk_optim

# Env
from environment import get_dataset_environment, get_model_environment

# Updater
from local import LocalUpdater
from local import LocalUpdate as LU


def main(args: Namespace) -> None:
    """
    Starting point
    """
    # Paths
    root: Path = args.save_root
    root.mkdir(parents=True, exist_ok=True)

    assets = root.joinpath('assets')
    assets.mkdir(parents=True, exist_ok=True)

    plots = root.joinpath('plots')
    plots.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if args.verbose:
        print(device)
    dataset_fs = open(str(root.joinpath('dataset-separation.txt')), 'a')

    # Save Parameter
    with open(str(root.joinpath('parameters.txt')), 'a') as f:
        f.write('Experiment Instruction\n')
        for key, value in vars(args).items():
            f.write(f'{key}\t{value}\n')

    shutil.copy(args.ds_config, root.joinpath(args.ds_config.name))
    cnf = read_py_config(args.ds_config)

    LocalUpdate = LocalUpdater[args.agg]

    # Dataset Split
    clients_ds, val_ds, _ = get_dataset_environment(args, cnf, dataset_fs)

    valid_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    # Server Model
    server_model = get_model_environment(args, cnf)
    server_model.to(device)

    with open(assets.joinpath('architecture.txt'), 'w') as f:
        total_params = sum(p.numel() for p in server_model.parameters())
        total_trainable_params = sum(p.numel() for p in server_model.parameters() if p.requires_grad)
        print(f'Total Parameters: {total_params}', file=f)
        print(f'Total Trainable Parameters: {total_trainable_params}', file=f)
        print(f'Total None Trainable Parameters: {total_params - total_trainable_params}', file=f)
        print(server_model, file=f)

    criterion = mk_loss(args)

    rnd_vl_acc = []
    rnd_vl_loss = []
    rnd_cl = []
    rnd_resume = 0
    if cnf['model'].get('checkpoint', None) is not None:
        state_dict = torch.load(cnf['model']['checkpoint'])
        server_model.load_state_dict(state_dict['model'])
        rnd_vl_acc = state_dict['rnd_vl_acc']
        rnd_vl_loss = state_dict['rnd_vl_loss']
        rnd_cl = state_dict['rnd_cl']
        rnd_resume = min(state_dict['rnd_resume'], len(rnd_cl))

    client_updater = [LocalUpdate(f'Client-{str(idx).zfill(3)}', args, server_model, device, dataset=cl_ds) for
                      idx, cl_ds in enumerate(clients_ds)]
    # Train
    for rnd_idx in range(rnd_resume, args.rounds):
        # Select clients
        if args.client_select == 'all':
            idx_users = list(range(args.n_clients))
        else:
            # Random
            m = max(int(args.client_frac * args.n_clients), 1)
            idx_users = np.random.choice(range(args.n_clients), m, replace=False)

        # Send Models
        for cl in client_updater:
            cl.set_parameter(server_model)

        # Train Client ith - Local update
        local_models = []
        local_weights = []
        total_samples = 0
        str_rnd = f"[{rnd_idx + 1}/{args.rounds}]"
        local_cl_res = []
        for cl_idx in idx_users:
            # Load Server model on Client ith
            cl_u: LU = client_updater[cl_idx]
            res = cl_u.train(str_rnd=str_rnd,cnf=cnf,args=args)
            local_weights.append(cl_u.train_samples)
            local_models.append(cl_u.model)
            total_samples += cl_u.train_samples
            local_cl_res.append({
                'name': cl_u.name,
                **res,
                **cl_u.train_time_cost
            })
        rnd_cl.append(local_cl_res)

        # Normalize the weights
        for i, w in enumerate(local_weights):
            local_weights[i] = w / total_samples

        # Aggregation
        assert len(local_models) > 0
        server_model = copy.deepcopy(local_models[0])
        for param in server_model.parameters():
            param.data.zero_()
        for w, client_model in zip(local_weights, local_models):
            for srv_param, cl_param in zip(server_model.parameters(), client_model.parameters()):
                srv_param.data += cl_param.data.clone() * w

        # Evaluate Server Model
        with torch.no_grad():
            val_loss = 0.0
            val_acc = 0.0
            val_iter = tqdm(valid_ld)
            str_rd = f"[{rnd_idx + 1}/{args.rounds}]"
            step = 0
            for i, data in enumerate(val_iter):
                str_st = f"[{i + 1}/{len(valid_ld)}]"
                data = tuple(fr.to(device) for fr in data)
                y = data[-1].view(-1)
                if cnf['model']['name'] == 'relationNet':
                    y = F.one_hot(y,5)
                    y = y.float()

                output = server_model(data)
                loss = criterion(output, y)
                val_loss += loss.item()

                if cnf['model']['name'] == 'relationNet':
                    batch_acc = metrics.accuracy_score(np.argmax(y.detach().cpu().numpy(),axis=-1),
                                                       np.argmax(output.detach().cpu().numpy(), axis=-1))
                else:
                    batch_acc = metrics.accuracy_score(y.detach().cpu().numpy(),
                                                       np.argmax(output.detach().cpu().numpy(), axis=-1))
                val_acc += batch_acc
                str_postfix = f'[Server Valid] Round: {str_rd} Step: {str_st} Batch Acc: {round(batch_acc, 3)} Acc: {round(val_acc / (i + 1), 3)} Batch Loss: {round(loss.item(), 3)} Loss: {round(val_loss / (i + 1), 3)}'
                show = OrderedDict({f'[Train]': str_postfix})
                val_iter.set_postfix(show)
                step = i
            rnd_vl_acc.append(val_acc / (step + 1))
            rnd_vl_loss.append(val_loss / (step + 1))

        # Checkpoint
        torch.save({
            'model': server_model.state_dict(),
            'rnd_resume': rnd_idx + 1,
            'rnd_vl_loss': rnd_vl_loss,
            'rnd_vl_acc': rnd_vl_acc,
            'rnd_cl': rnd_cl
        }, root.joinpath('cpt.latest.pt'))

    dataset_fs.close()
    # Plot
    colormap = plt.cm.gist_ncar
    plt.rcParams.update({'font.size': 140})
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, args.epochs))))

    all_rnd = list(range(len(rnd_cl)))
    for idx, cl_res in tqdm(zip(all_rnd[-5:], rnd_cl[-5:]), total=len(rnd_cl[-5:])):
        plt_save = plots.joinpath(f'{str(idx+1).zfill(4)}.jpg')
        legend_labels = []
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(200, 80))
        for cl in cl_res:
            legend_labels.append(cl['name'])
            ax_acc.plot(np.arange(args.epochs), cl['accuracy'], linewidth=7.0)
            ax_loss.plot(np.arange(args.epochs), cl['loss'], linewidth=7.0)
        ax_loss.legend(legend_labels)
        ax_loss.set_title("Loss")
        ax_acc.legend(legend_labels, )
        ax_acc.set_title("Accuracy")
        fig.savefig(plt_save)
        plt.clf()

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(200, 80))
    ax_acc.plot(np.arange(args.rounds), rnd_vl_acc, linewidth=7.0)
    ax_acc.set_title("Accuracy")
    ax_loss.plot(np.arange(args.rounds), rnd_vl_loss, linewidth=7.0)
    ax_loss.set_title("Loss")
    fig.savefig(root.joinpath('server.jpg'))
    plt.clf()


if __name__ == '__main__':
    parser = ArgumentParser()

    # Path
    parser.add_argument('--save_root', help='Save root path', type=Path, default='./experimental/run1')
    parser.add_argument('--ds-config', dest='ds_config', help='Dataset Configuration (.py)', type=Path,
                        default='./config/rn-cifar.py')
    parser.add_argument('--verbose', help='Log more details', action='store_true', default=True)
    parser.add_argument('--seed', help='Random State', type=int, default=2023)
    parser.add_argument('--device', help='Device Cuda, Cpu', type=str, default='cuda', choices=['cpu', 'cuda'])

    # Hyperparameter
    parser.add_argument('--batch', help='Batch size', type=int, default=4)
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-4)

    parser.add_argument('--epochs', help='No.Epochs', type=int, default=100)
    parser.add_argument('--rounds', help='No.Rounds', type=int, default=50)
    parser.add_argument('--loss', help='Loss function', type=str, default='MSE')
    parser.add_argument('--agg', help='Aggregation Strategy', type=str, choices=list(LocalUpdater.keys()),
                        default='FedAvg')

    # Optimizer
    parser.add_argument('--opt', help='Optimizer type', default='SGD', choices=list(RegisterOptim.keys()))
    parser.add_argument('--opt-weight-decay', '--opt_weight_decay', help='Optimizer weight decay', type=float,
                        default=0.)
    parser.add_argument('--opt-momentum', '--opt_momentum', help='Optimizer momentum', type=float, default=0.)
    parser.add_argument('--opt-prox-mu', '--opt_prox_mu', help='FedProx opt mu', type=float, default=0.)

    # Augmentation

    # Color
    parser.add_argument('--color', help='Enable ColorJitter', action='store_true', default=True)
    parser.add_argument('--color-br', dest='color_br', help='Color Brightness Beta', type=float, default=.4)
    parser.add_argument('--color-co', dest='color_co', help='Color Contrast Alpha', type=float, default=0.4)
    parser.add_argument('--color-st', dest='color_st', help='Color Stature', type=float, default=0.4)

    # Crop
    parser.add_argument('--img-size', dest='img_size', help='Image Size', type=int, default=80)
    parser.add_argument('--crop-x', '--crop_x', help='Crop on x axis', type=float, default=1.)
    parser.add_argument('--crop-y', '--crop_y', help='Crop on y axis', type=float, default=1.)

    # Horizontal Flip
    parser.add_argument('--h-flip', dest='h_flip', help='Enable Horizontal Flip', action='store_true', default=True)
    parser.add_argument('--h-flip-p', dest='h_flip_p', help='Horizontal Flip Probability', type=float, default=.5)

    # Vertical Flip
    parser.add_argument('--v-flip',dest='v_flip',help='Enable Vertical Flip',action='store_true',default=True)
    parser.add_argument('--v-flip-p',dest='v_flip_p',help='Vertical Flip Probability',type=float,default=.5)

    # Rotation
    parser.add_argument('--rotate', help='Enable rotation', action='store_true', default=True)
    parser.add_argument('--rotate-angle',dest='rotate_angle',help='Rotation Angle', type=float,default=25.)

    # Shear
    parser.add_argument('--shear', help='Enable shearing', action='store_true', default=True)

    # Dataset Split
    parser.add_argument('--iid', help='Enable iid', action='store_true', default=True)
    parser.add_argument('--balance', help='Enable Balance for IID', action='store_true', default=True)
    parser.add_argument('--alpha', help='Dirichlet alpha', type=float, default=1e-1)

    # Client
    parser.add_argument('--client-select', '--client_select', help='Select Client', type=str, default='random',
                        choices=['random', 'all'])
    parser.add_argument('--client-frac', '--client_frac', help='The fraction of clients', type=float, default=.5)
    parser.add_argument('--n-clients', '--n_clients', help='No.Clients', type=int, default=10)

    main(args=parser.parse_args())
