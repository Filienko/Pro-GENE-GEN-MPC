import os, sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import random
import pickle
import shutil

from opacus import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler

sys.path.append("../..")
from model import CVAE
from utils import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        "-name",
        type=str,
        required=True,
        help="path for storing the checkpoint",
    )
    parser.add_argument(
        "--random_seed", "-s", type=int, default=1000, help="random seed"
    )
    parser.add_argument(
        "--split_seed", type=int, default=1000, help="random seed for data splitting"
    )
    parser.add_argument(
        "--device_type", "-d", type=str, default="cuda", help="type of device"
    )

    ### data config
    parser.add_argument(
        "--dataset",
        "-data",
        type=str,
        default="aml",
        choices=["aml", "bulk_aml"],
        help="dataset name",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="standarize",
        choices=["standarize", "minmax", "none", "discretize"],
        help="pre-processing scheme",
    )
    parser.add_argument(
        "--preprocess_arg",
        metavar="KEY=VALUE",
        default={},
        nargs="+",
        action=ParseDict,
        help="set the arguments for preprocess using key-value pairs",
    )
    parser.add_argument(
        "--if_filter_x",
        type=str2bool,
        default=True,
        help="If filter the features using the 1000 landmark genes",
    )

    ### model training config
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--beta", type=float, default=0.001, help="weight for kl loss")
    parser.add_argument(
        "--num_iters", "-iters", type=int, default=10000, help="number of iters"
    )
    parser.add_argument("--z_dim", "-z_dim", type=int, default=64, help="z_dim")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--transform",
        type=str,
        default="none",
        choices=["exp", "none", "tahn", "sigmoid", "relu"],
        help="output transform function",
    )

    ### eval setting
    parser.add_argument(
        "--eval_model_type",
        type=str,
        default="default",
        choices=["default", "logistic", "svc_l1"],
    )
    parser.add_argument("--eval_iter", type=int, default=500)
    parser.add_argument("--eval_frac", type=float, default=0.5, help="eval fraction")
    parser.add_argument("--test_frac", type=float, default=0.2, help="test fraction")
    parser.add_argument(
        "--num_samples_ratio",
        type=int,
        default=1,
        help="num of generated samples(ratio to ori size)",
    )
    parser.add_argument(
        "--if_uniform_y", type=str2bool, default=False, help="If sample y uniformly"
    )
    parser.add_argument(
        "--if_valid",
        type=str2bool,
        default=False,
        help="If further split test dataset to test/valid",
    )

    parser.add_argument(
        "--if_verbose",
        type=str2bool,
        default=False,
        help="To evaluate during training",
    )

    ### save config
    parser.add_argument(
        "--if_save_model",
        "-save",
        type=str2bool,
        default=False,
        help="If save the final model",
    )
    parser.add_argument(
        "--if_save_data",
        type=str2bool,
        default=False,
        help="If save the fake and real data",
    )
    parser.add_argument(
        "--if_resume", type=str2bool, default=False, help="If restore the training"
    )
    parser.add_argument(
        "--if_evaluate", type=str2bool, default=False, help="If only eval the model"
    )
    parser.add_argument(
        "--if_impute", type=str2bool, default=False, help="If impute the missing values"
    )

    ### DP related arguments
    parser.add_argument(
        "--enable_privacy",
        type=str2bool,
        default=False,
        help="Enable private data generation",
    )
    parser.add_argument(
        "--target_epsilon", type=float, default=10, help="Epsilon DP parameter"
    )
    parser.add_argument(
        "--target_delta", type=float, default=1e-5, help="Delta DP parameter"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Gaussian noise variance multiplier. A larger sigma will make the model train for longer epochs for the same privacy budget",
    )
    parser.add_argument(
        "--max_norm",
        type=float,
        default=0.1,
        help="The coefficient to clip the gradients before adding noise for private SGD training",
    )

    args = parser.parse_args()
    return args


DATA_DIR = "../../data/"


def check_args(args):
    """
    check and store the arguments as well as set up the save_dir
    :param args: arguments
    :return:
    """
    ## set up save_dir
    save_dir = os.path.join(
        os.path.dirname(__file__), "results", args.dataset, args.exp_name
    )
    mkdir(save_dir)

    ## store the parameters
    if not args.if_evaluate:
        if args.if_resume:  ## check parameter matching if resuming from checkpoint
            try:
                stored_param_dict = pickle.load(
                    open(os.path.join(save_dir, "params.pkl"), "rb")
                )
                for k, v in stored_param_dict.items():
                    new_v = vars(args)[k]
                    if v != new_v:
                        print(
                            f"!!!parameter '{k}' does not match, rewrite from '{v}' to '{new_v}'"
                        )
            except:
                pass

        ## store the parameters
        with open(os.path.join(save_dir, "params.txt"), "w") as f:
            for k, v in vars(args).items():
                f.writelines(k + ":" + str(v) + "\n")
                print(k + ":" + str(v))
        pickle.dump(
            vars(args), open(os.path.join(save_dir, "params.pkl"), "wb"), protocol=2
        )

    else:  ## run evaluation mode
        mkdir(save_dir + "/eval")
        mkdir(save_dir + "/samples")

    ## store this script
    shutil.copy(os.path.realpath(__file__), save_dir)
    return args, save_dir


def main():
    ### config
    args, save_dir = check_args(parse_arguments())

    ### Random seed
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    ### set device
    device = torch.device(args.device_type)

    ### CUDA
    if device.type == "cuda":
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.manual_seed_all(args.random_seed)
    else:
        use_mps = torch.backends.mps.is_available()

    ### Paths
    dset_dir = DATA_DIR
    save_path = os.path.join(save_dir, "model.pkt")

    ### Load data
    if "aml" in args.dataset:
        dset = AML(
            dset_dir,
            preprocess=args.preprocess,
            if_filter_x=args.if_filter_x,
            dataset_name=args.dataset,
            **args.preprocess_arg,
        )
        eval_model_type = (
            "svc_l1" if args.eval_model_type == "default" else args.eval_model_type
        )

    x_dim, y_dim = dset.get_dim()
    print("x_dim: %d, y_dim: %d" % (x_dim, y_dim))

    ### Set up Evaluation model
    if args.test_frac > 0:
        train_x, train_y, test_x, test_y = dset.train_test(
            k=args.split_seed, test_fraction=args.test_frac
        )
    else:
        test_x, test_y = dset.dset, dset.anno
    print("size of train dataset: %d" % len(dset))
    eval_subset = int(args.eval_frac * len(test_x))

    if args.if_valid:
        valid_x, valid_y = test_x[:eval_subset], test_y[:eval_subset]
        test_x, test_y = test_x[eval_subset:], test_y[eval_subset:]

        print("size of test dataset: %d" % len(test_x))
        print("size of valid dataset: %d" % len(valid_x))
    else:
        valid_x, valid_y = test_x, test_y
        print("test dataset size = valid dataset size: %d" % len(valid_x))

    if args.if_evaluate and args.if_verbose:
        ref_acc = Eval(valid_x, valid_y, model_type=eval_model_type).efficacy(
            train_x, train_y, seed=args.random_seed
        )
        print("=" * 100)
        print(
            f"reference accuracy={ref_acc}, (classification on real data with dim={x_dim})"
        )

    ### Specify parameters for DP dataloader
    if args.if_evaluate:
        loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False)
    else:
        if args.enable_privacy:
            generator = None
            kwargs = {"num_workers": 0, "pin_memory": True}
            loader = DataLoader(
                dset,
                generator=generator,
                batch_sampler=UniformWithReplacementSampler(
                    num_samples=len(dset),
                    sample_rate=args.batch_size / len(dset),
                    generator=generator,
                ),
            )
        else:
            loader = DataLoader(dset, batch_size=args.batch_size, shuffle=True)

    train_loader = inf_train_gen(loader)

    ### Set up VAE models and optimizer
    model = CVAE(
        x_dim=x_dim,
        y_dim=y_dim,
        z_dim=args.z_dim,
        beta=args.beta,
        transform=args.transform,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    ### Set up logger (for storing the results) and restore checkpoint
    min_loss = 1e9
    title = "cvae"
    if args.if_resume or args.if_evaluate:
        print("==> Resuming from checkpoint..")
        checkpoint = torch.load(
            os.path.join(save_dir, "checkpoint.pkl")
        )  # load the checkpoint
        iter_start = checkpoint["iter"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["opt_state_dict"])
        logger = Logger(os.path.join(save_dir, "log.txt"), title=title, resume=True)
        print("iterstart: {}".format(iter_start))
    else:
        iter_start = 1
        logger = Logger(os.path.join(save_dir, "log.txt"), title=title)
        logger.set_names(
            [
                "Iter",
                "Loss",
                "Loss Rec",
                "Loss KL",
                "Real Acc",
                "Fake Acc",
                "Real Distance",
                "Fake Distance",
                "epsilon",
            ]
        )

    ### If only evaluate
    if args.if_evaluate:
        try:
            model.load_state_dict(torch.load(save_path))  # load model
        except:
            print("unable to load model.")
            pass

        ### eval
        fake_data_all, fake_label_all = [], []
        for i in range(args.num_samples_ratio):
            fake_data_, fake_label_ = model.sample(
                len(dset), loader, args.if_uniform_y, device=device
            )
            fake_data_all.append(fake_data_)
            fake_label_all.append(fake_label_)
            fake_data = np.concatenate(fake_data_all)
            fake_label = np.concatenate(fake_label_all)

            ### Efficacy
            # train on real test data and score on fake data
            print("Eval model: {}".format(eval_model_type))
            real_acc = Eval(fake_data, fake_label, model_type=eval_model_type).efficacy(
                test_x, test_y
            )

            # train on fake data and score on real test data
            fake_acc = Eval(test_x, test_y, model_type=eval_model_type).efficacy(
                fake_data, fake_label
            )

            ### Knearest Neigbors
            real_distance = Eval(test_x, test_y).kneighbors(train_x)
            fake_distance = Eval(test_x, test_y).kneighbors(fake_data)

            print("=" * 100)
            print("ratio= " + str(i))
            print("real acc: ", str(real_acc))
            print("fake acc: ", str(fake_acc))
            print("real distance: ", str(real_distance))
            print("fake distance: ", str(fake_distance))

            write_csv(
                os.path.join(save_dir, f"eval/eval_s{args.random_seed}.csv"),
                "ratio= " + str(i) + " uniform" if args.if_uniform_y else "",
                [real_acc, fake_acc, real_distance, fake_distance],
                ["real acc", "fake acc", "real distance", "fake distance"],
            )

        if args.if_save_data:
            save_data_csv(
                os.path.join(
                    save_dir, f"samples/k{args.split_seed}_s{args.random_seed}.csv"
                ),
                dset._inverse_transform(fake_data),
                fake_label,
                dset.column_names,
            )
            original_string_labels = [dset.label_map[int(l)] for l in fake_label]
            save_data_csv(
                os.path.join(
                    save_dir, f"samples/k{args.split_seed}_s{args.random_seed}_w_label.csv"
                ),
                dset._inverse_transform(fake_data),
                original_string_labels, # Use the string labels here
                dset.column_names,
            )

        return

    ### DP optimizer
    if args.enable_privacy:
        # model = copy.deepcopy(model).to(device) # shadow model for obtaining DP gradients
        model = model.to(device)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        if args.if_resume:
            opt.load_state_dict(checkpoint["opt_state_dict"])
        k = args.num_iters
        epsilon = args.target_epsilon
        delta = args.target_delta
        noise_multiplier = compute_sigma(epsilon, args.batch_size / len(dset), k, delta)
        print(
            f"eps,delta = ({epsilon},{delta}) ==> Noise level sigma=", noise_multiplier
        )

        sigma = noise_multiplier if args.sigma is None else args.sigma
        privacy_engine = PrivacyEngine(
            model,
            sample_rate=args.batch_size / len(dset),
            sample_size=len(dset),
            batch_size=args.batch_size,
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=sigma,
            max_grad_norm=args.max_norm,
        )
        privacy_engine.attach(opt)

    for iters in range(iter_start, args.num_iters + 1):
        ## Update step
        data_x, data_y = next(train_loader)
        data_x = data_x.to(device)
        data_y = one_hot_embedding(data_y, num_classes=y_dim, device=device)

        model.train()
        model.zero_grad()
        output = model.compute_loss(data_x, data_y, verbose=False)
        loss = output["loss"]
        loss.backward()
        opt.step()

        ### Evaluation
        if args.if_evaluate and args.if_verbose:
            if iters % args.eval_iter == 0 or (iters < 200 and iters % 50 == 0):
                fake_data, fake_label = model.sample(
                    len(dset), loader, args.if_uniform_y, device=device
                )
                real_acc = Eval(
                    fake_data, fake_label, model_type=eval_model_type
                ).efficacy(valid_x, valid_y, seed=args.random_seed)
                fake_acc = Eval(valid_x, valid_y, model_type=eval_model_type).efficacy(
                    fake_data,
                    fake_label,
                    subset=eval_subset,
                    seed=args.random_seed,
                )

                fake_distance = Eval(test_x, test_y).kneighbors(
                    fake_data, subset=eval_subset
                )
                real_distance = Eval(test_x, test_y).kneighbors(
                    train_x, subset=eval_subset
                )

                if args.enable_privacy:
                    epsilon, best_alpha = opt.privacy_engine.get_privacy_spent(
                        args.target_delta
                    )
                else:
                    epsilon = np.inf

                ### Print and store results in log file
                print(
                    f"Iter {iters}, loss: {loss.item()}, real acc: {real_acc}, fake acc: {fake_acc}, real distance: {real_distance}, fake distance: {fake_distance}"
                )
                logger.append(
                    [
                        iters,
                        loss.item(),
                        output["rec_loss"].item(),
                        output["kl_loss"].item(),
                        real_acc,
                        fake_acc,
                        real_distance,
                        fake_distance,
                        epsilon,
                    ]
                )

        ### Save/update the checkpoint during training
        save_dict = {
            "iter": iters,
            "model_state_dict": model.state_dict(),
            "opt_state_dict": opt.state_dict(),
        }
        torch.save(save_dict, os.path.join(save_dir, "checkpoint.pkl"))
        model.state_dict()

        ### Save the best model
        if args.if_save_model:
            if loss < min_loss:
                min_loss = loss
                torch.save(model.state_dict(), save_path)
        if args.if_verbose:
            ### Plot results during training
            logger.plot(["Loss Rec"], x="Iter")
            savefig(os.path.join(save_dir, "loss_rec.png"))

            logger.plot(["Loss"], x="Iter")
            savefig(os.path.join(save_dir, "loss.png"))

            logger.plot(["Real Acc", "Fake Acc"], x="Iter")
            savefig(os.path.join(save_dir, "acc.png"))

            logger.plot(["Real Distance", "Fake Distance"], x="Iter")
            savefig(os.path.join(save_dir, "distance.png"))
    print("Plotting Results")
    if args.if_verbose:
        ### Plot all results
        logger.close()
        logger.plot(["Loss Rec"], x="Iter")
        savefig(os.path.join(save_dir, "loss_rec.png"))

        logger.plot(["Loss"], x="Iter")
        savefig(os.path.join(save_dir, "loss.png"))

        logger.plot(["epsilon"], x="Iter")
        savefig(os.path.join(save_dir, "epsilon.png"))

        logger.plot(["Real Acc", "Fake Acc"], x="Iter")
        savefig(os.path.join(save_dir, "acc.png"))

        logger.plot(["Real Distance", "Fake Distance"], x="Iter")
        savefig(os.path.join(save_dir, "distance.png"))


if __name__ == "__main__":
    main()
