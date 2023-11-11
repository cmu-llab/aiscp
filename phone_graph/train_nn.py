import numpy as np
from scipy import linalg as la
from scipy import spatial

import argparse
import pprint
import pdb
import pickle as pkl
import re
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import trange
from tqdm import tqdm


def seed_everything(seed=sum(bytes(b'llab'))):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class FeatChangeDNN(nn.Module): # classifies NPI outputs
    def __init__(self, in_dim, out_dim):

        super(FeatChangeDNN, self).__init__()

        print(f"Initializing 4-layer model with in dim {in_dim} and out dim {out_dim}", flush=True)
        
        increment = (out_dim - in_dim) // 7
        dim1 = in_dim + increment
        dim2 = dim1 + increment
        dim3 = dim2 + increment
        dim4 = dim3 + increment

        dim7 = out_dim - increment
        dim6 = dim7 - increment
        dim5 = dim6 - increment

        self.model = nn.Sequential(
                                    nn.Linear(in_dim, dim1),
                                    nn.ReLU(),
                                    nn.Linear(dim1, dim2),
                                    nn.ReLU(),
                                    nn.Linear(dim2, dim3),
                                    nn.ReLU(),
                                    nn.Linear(dim3, dim4),
                                    nn.ReLU(),
                                    nn.Linear(dim4, dim5),
                                    nn.ReLU(),
                                    nn.Linear(dim5, dim6),
                                    nn.ReLU(),
                                    nn.Linear(dim6, dim7),
                                    nn.ReLU(),
                                    nn.Linear(dim7, out_dim),
                                    nn.Sigmoid()
                                    )

    def forward(self, x):
        return self.model(x)

class FeatChangeDResNN(nn.Module): # classifies NPI outputs
    def __init__(self, in_dim, out_dim):

        super(FeatChangeDResNN, self).__init__()

        print(f"Initializing 4-layer model with in dim {in_dim} and out dim {out_dim}", flush=True)

        self.l1 = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU()
                )
        self.l2 = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU()
                )
        self.l3 = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU()
                )
        self.l4 = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU()
                )
        self.l5 = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU()
                )
        self.l6 = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU()
                )
        self.l7 = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU()
                )
        self.l8 = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU()
                )
        self.l9 = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU()
                )
        self.l10 = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU()
                )
        self.l11 = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU()
                )
        self.l12 = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU()
                )
        self.l13 = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU()
                )
        self.l14 = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU()
                )
        self.l15 = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU()
                )
        self.l16 = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Sigmoid()
                )

    def forward(self, x):

        x1 = self.l1(x)
        x2 = self.l2(x1) + x1
        x3 = self.l3(x2) + x2
        x4 = self.l4(x3) + x3
        x5 = self.l5(x4) + x4
        x6 = self.l6(x5) + x5
        x7 = self.l7(x6) + x6
        x8 = self.l8(x7) + x7
        x9 = self.l9(x8) + x8
        x10 = self.l10(x9) + x9
        x11 = self.l11(x10) + x10
        x12 = self.l12(x11) + x11
        x13 = self.l13(x12) + x12
        x14 = self.l14(x13) + x13
        x15 = self.l7(x14) + x14 + x
        x16 = self.l16(x15)

        return x16

class FeatChangeResNN(nn.Module): # classifies NPI outputs
    def __init__(self, in_dim, out_dim):

        super(FeatChangeResNN, self).__init__()

        print(f"Initializing 4-layer model with in dim {in_dim} and out dim {out_dim}", flush=True)

        self.l1 = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU()
                )
        self.l2 = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU()
                )
        self.l3 = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU()
                )
        self.l4 = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU()
                )
        self.l5 = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU()
                )
        self.l6 = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU()
                )
        self.l7 = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU()
                )
        self.l8 = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Sigmoid()
                )

    def forward(self, x):

        x1 = self.l1(x)
        x2 = self.l2(x1) + x1
        x3 = self.l3(x2) + x2
        x4 = self.l4(x3) + x3
        x5 = self.l5(x4) + x4
        x6 = self.l6(x5) + x5
        x7 = self.l7(x6) + x6 + x
        x8 = self.l8(x7)

        return x8

class FeatChangeNN(nn.Module): # classifies NPI outputs
    def __init__(self, in_dim, out_dim):
        
        super(FeatChangeNN, self).__init__()
        
        print(f"Initializing 4-layer model with in dim {in_dim} and out dim {out_dim}", flush=True)

        increment = (out_dim - in_dim) // 3
        dim1 = in_dim + increment
        dim2 = dim1 + increment
        dim3 = dim2 + increment

        self.model = nn.Sequential(
                                    nn.Linear(in_dim, dim1),
                                    nn.ReLU(),
                                    nn.Linear(dim1, dim2),
                                    nn.ReLU(),
                                    nn.Linear(dim2, dim3),
                                    nn.ReLU(),
                                    nn.Linear(dim3, out_dim),
                                    nn.Sigmoid()
                                    )

    def forward(self, x):
        return self.model(x)

class FeatChangeMatrix(nn.Module): # classifies NPI outputs
    def __init__(self, in_dim, out_dim):

        super(FeatChangeMatrix, self).__init__()

        print(f"Initializing single-layer model with in dim {in_dim} and out dim {out_dim}", flush=True)

        self.model = nn.Sequential(
                                    nn.Linear(in_dim, out_dim),
                                    nn.Sigmoid()
                                    )

    def forward(self, x):
        return self.model(x)


def train_nn(args):
    """
    Train transformation matrix on training set
    """

    print("Reading vector bitext...", flush=True)
    global VEC_BITEXT # list of tuples of torch tensors
    with open(args.vec_bitext, 'rb') as f:
        VEC_BITEXT = pkl.load(f)

    VEC_BITEXT = [(torch.from_numpy(tup[0]).float(), torch.from_numpy(tup[1]).float()) for tup in VEC_BITEXT]

    # shuffle keys with random permutation
    random.shuffle(VEC_BITEXT)

    print("Read vector bitext, length = {}".format(len(VEC_BITEXT)), flush=True)

    # Data for training
    training_testing_amount = int(args.train_test_split * len(VEC_BITEXT))
    TRAIN_BITEXT = VEC_BITEXT[:training_testing_amount]
    training_amount = int(args.train_test_split * training_testing_amount)
    train_data = TRAIN_BITEXT[:training_amount]
    test_data = TRAIN_BITEXT[training_amount:]

    in_dim, out_dim = args.in_dim, args.out_dim
    if not in_dim:
        in_dim = len(VEC_BITEXT[0][0])
    if not out_dim:
        out_dim = len(VEC_BITEXT[0][1])

    # Initialize model
    #emb_dim = len(VEC_DICT_SRC[list(VEC_DICT_SRC.keys())[0]])
    if args.model_type == '1L':
        nn_model = FeatChangeMatrix(in_dim, out_dim).to(args.device)
    elif args.model_type == '4L':
        nn_model = FeatChangeNN(in_dim, out_dim).to(args.device)
    elif args.model_type == '8L':
        nn_model = FeatChangeDNN(in_dim, out_dim).to(args.device)
    elif args.model_type == '8res':
        nn_model = FeatChangeResNN(in_dim, out_dim).to(args.device)
    elif args.model_type == '16res':
        nn_model = FeatChangeDResNN(in_dim, out_dim).to(args.device)

    # Initialize objective and optimizer
    #matrix_objective = nn.MSELoss()
    nn_objective = nn.BCELoss()
    nn_optimizer = optim.Adam(nn_model.parameters(), lr=args.learning_rate)

    # make training data set and data loader
    train_loader = DataLoader(train_data, batch_size=args.batch_size, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, pin_memory=True)

    # define loop
    loop = tqdm(total=len(train_loader), position=0, leave=False)

    for epoch in range(args.num_epochs):
        # loop through data set
        for batch, (src_vec, tgt_vec) in enumerate(train_loader):
            src_vec, tgt_vec = src_vec.to(args.device), tgt_vec.to(args.device)

            nn_optimizer.zero_grad()

            # predict translation
            pred_vec = nn_model(src_vec)
            # calculate loss
            loss = nn_objective(pred_vec, tgt_vec)
            loss_item = loss.item() # for collecting and printing
            # backprop and step
            loss.backward()
            nn_optimizer.step()

            # report current state to terminal
            loop.set_description('epoch:{}, loss:{}'.format(epoch, loss_item))
            loop.update(1)

        if epoch % args.test_freq == 0:
            test_losses = []
            for test_batch, (test_src_vec, test_tgt_vec) in enumerate(test_loader):
                test_src_vec, test_tgt_vec = test_src_vec.to(args.device), test_tgt_vec.to(args.device)

                # predict translation
                test_pred_vec = nn_model(test_src_vec)
                # calculate loss
                test_loss = nn_objective(test_pred_vec, test_tgt_vec)
                test_loss_item = test_loss.item() # for collecting and printing
                test_losses.append(test_loss_item)

            avg_test_loss = np.mean(test_losses)
            print("")
            print("Average test loss for epoch {}: {}".format(epoch, avg_test_loss), flush=True)

    # wrap up
    print("")
    print("training finished",flush=True)
    return nn_model


def eval_nn(args, nn_model):

    # FUNCTION DEPRECATED

    # Data for eval
    training_testing_amount = int(args.train_test_split * len(VEC_BITEXT))
    EVAL_BITEXT = VEC_BITEXT[training_testing_amount:]

    print("Beginning evaluation...", flush=True)

    print_every_x_percent = 10
    times_to_print = 100 // print_every_x_percent
    print_every = len(EVAL_BITEXT) // times_to_print
    #pdb.set_trace()
    
    num_correct = 0
    num_evaluated = 0
    original_vec_idx = training_testing_amount
    for src_vec, tgt_vec in EVAL_BITEXT:
        # send to cuda
        src_vec = src_vec.to(args.device)
        src_word, tgt_word = ACTUAL_WORD_BITEXT[original_vec_idx]

        pred_vec = nn_model(src_vec).detach()

        # Now find the closest vector in the space to the pred_vec
        all_tgt_tokens = list(VEC_DICT_TGT.keys()) # FIXME: This line may cause problems: give advantage to embeddings w/ small vocab
        all_tgt_vectors = [VEC_DICT_TGT[tok] for tok in all_tgt_tokens]
        proposed_tgt_vec, proposed_tgt_word = find_closest_point(pred_vec.cpu().squeeze().numpy(), all_tgt_tokens, all_tgt_vectors) # FIXME: This uses cosine dist not the objective MSE dist
        #print("{} =?= {}".format(tgt_word, proposed_tgt_word.decode()), flush=True)

        if np.allclose(np.array(proposed_tgt_vec), np.array(tgt_vec.squeeze())):
            num_correct += 1
        num_evaluated += 1

        if num_evaluated % print_every == 0:
            percent_done = round((num_evaluated / len(EVAL_BITEXT))*100)
            print("{}%".format(percent_done), end="\t", flush=True)

        original_vec_idx += 1

    score = num_correct / num_evaluated

    print("\n")
    print("Score for {}: {}".format(args.vec_file_src, score),flush=True)

    return score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--vec_bitext",type=str,help="file with training data as List[Tuple[torch.tensor]]",required=True)
    parser.add_argument("--save_file",type=str,help="file to write model to",required=True)
    parser.add_argument("--in_dim",type=int,default=None,help="length of input")
    parser.add_argument("--out_dim",type=int,default=None,help="length of output")
    # Training hyperparams
    parser.add_argument("--num_epochs",type=int,default=25,help="number of epochs for matrix training")
    parser.add_argument("--test_freq",type=int,default=2,help="how often (every which epoch) do we want to run the testing loop?")
    parser.add_argument("--batch_size",type=int,default=5,help="batch size throughout matrix training")
    parser.add_argument("--learning_rate",type=float,default=.001,help="learning rate for optimizer")
    parser.add_argument("--train_test_split",type=float,default=0.9,help="ratio of total data to be used in training")
    parser.add_argument("--gpu_num",type=int,default=None,help="GPU to use")
    parser.add_argument("--seed",type=str,default=sum(bytes(b'llab')),help="seed for randomizing training data")
    parser.add_argument("--model_type",choices=["1L", "4L", "8L", "8res", "16res"],help="Depth of NN")
    args = parser.parse_args()

    if args.gpu_num is not None:
        args.device = torch.device("cuda:{}".format(args.gpu_num))
    else:
        if torch.cuda.is_available():
            args.device = torch.device("cuda")
        else:
            args.device = torch.device("cpu")
            print("WARNING: Using CPU since no GPU available", flush=True)

    seed_everything(seed=args.seed)

    model = train_nn(args)

    torch.save(model, args.save_file)
    print("Saved model to", args.save_file)

    """
    sample run:
    python3 train_nn.py --vec_bitext path/to/bitext/pkl --save_file path/to/save/model/to
    """
