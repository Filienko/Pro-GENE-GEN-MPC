import os, sys
import numpy as np
import pandas as pd
import argparse
import pickle
import shutil

sys.path.append("../..")
from model import Private_PGM
from mpc_model import MPC_PrivatePGM
from utils import *

DATA_DIR = "../../data/"


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
        "--num_iters", "-iters", type=int, default=10000, help="number of iters"
    )

    ### data config
    parser.add_argument(
        "--dataset",
        "-data",
        type=str,
        default="aml",
        choices=["aml", "mouse", "bulk_aml"],
        help="dataset name",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="discretize",
        choices=["discretize"],
        help="pre-processing scheme (fixed to be discretize)",
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
        action="store_true",
        default=False,
        help="If filter the features",
    )

    ### eval setting
    parser.add_argument(
        "--num_samples_ratio",
        type=int,
        default=1,
        help="num of generated samples(ratio to ori size)",
    )
    parser.add_argument(
        "--eval_model_type",
        type=str,
        default="default",
        choices=["default", "logistic", "svc_l1"],
    )
    parser.add_argument("--eval_frac", type=float, default=0.2, help="eval fraction")
    parser.add_argument("--test_frac", type=float, default=0.2, help="test fraction")
    parser.add_argument(
        "--if_valid",
        type=str2bool,
        default=False,
        help="If further split test dataset to test/valid",
    )

    ### save config
    parser.add_argument(
        "--if_save_model",
        "-save",
        action="store_true",
        default=False,
        help="If save the final model",
    )
    parser.add_argument(
        "--if_resume",
        action="store_true",
        default=False,
        help="If restore the training",
    )
    parser.add_argument(
        "--if_evaluate",
        action="store_true",
        default=False,
        help="If only eval the model",
    )

    ### Private-pgm setting
    # subparsers = parser.add_subparsers(help="generative model type", dest="model")
    # privacy_parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--enable_privacy", action="store_true", help="Enable private data generation"
    )
    parser.add_argument(
        "--target_epsilon",
        type=float,
        default=8,
        help="Epsilon differential privacy parameter",
    )
    parser.add_argument(
        "--target_delta",
        type=float,
        default=1e-5,
        help="Delta differential privacy parameter",
    )

    ### MPC setting
    parser.add_argument(
        "--use_mpc",
        action="store_true",
        default=False,
        help="Use MPC-based Private PGM instead of centralized version",
    )
    parser.add_argument(
        "--mpc_path",
        type=str,
        default=None,
        help="Path to MP-SPDZ installation (default: /opt/MP-SPDZ or $MPC_PATH)",
    )
    parser.add_argument(
        "--mpc_protocol",
        type=str,
        default="ring",
        choices=["ring", "semi2k", "mascot"],
        help="MPC protocol to use (default: ring)",
    )
    # parser_pgm = subparsers.add_parser('private-pgm', parents=[privacy_parser])
    args = parser.parse_args()
    return args


def check_args(args):
    """
    check and store the arguments as well as set up the save_dir
    :param args: arguments
    :return:
    """
    ## set up save_dir
    save_dir = os.path.join(os.path.dirname(__file__), "results", args.exp_name)
    # save_dir = os.path.join(SAVE_DIR % (args.dataset, args.model_architecture), args.exp_name)
    mkdir(save_dir)

    ## store the parameters
    if not args.if_evaluate:
        with open(os.path.join(save_dir, "params.txt"), "w") as f:
            for k, v in vars(args).items():
                f.writelines(k + ":" + str(v) + "\n")
                print(k + ":" + str(v))
        pickle.dump(
            vars(args), open(os.path.join(save_dir, "params.pkl"), "wb"), protocol=2
        )

    ## store this script
    shutil.copy(os.path.realpath(__file__), save_dir)
    return args, save_dir


def main():
    ### config
    args, save_dir = check_args(parse_arguments())
    mkdir(save_dir + "/samples")

    ### Load data
    dset_dir = DATA_DIR
    if "aml" in args.dataset:
        dset = AML(
            dset_dir,
            preprocess=args.preprocess,
            if_filter_x=args.if_filter_x,
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
    print("Class distribution in training data:", pd.Series(train_y).value_counts())
    print("size of train dataset: %d" % len(dset))
    eval_subset = int(args.eval_frac * len(test_x))
    if args.if_valid:
        valid_x, valid_y = test_x[:eval_subset], test_y[:eval_subset]
        test_x, test_y = test_x[eval_subset:], test_y[eval_subset:]
        print("size of test dataset: %d" % len(test_x))
        print("size of valid dataset: %d" % len(valid_x))
    else:
        valid_x, valid_y = test_x, test_y
    ref_acc = Eval(valid_x, valid_y, model_type=eval_model_type).efficacy(
        train_x, train_y, seed=args.random_seed
    )
    print("=" * 100)
    print(
        f"reference accuracy={ref_acc}, (classification on real data with dim={x_dim})"
    )

    ### Convert to Dataframe
    merged = np.hstack((np.expand_dims(train_y, -1), train_x))
    concat_columns = np.concatenate((["label"], dset.column_names))
    train_dataframe = pd.DataFrame(merged, columns=concat_columns)

    data_columns = dset.column_names
    config = {}
    for col in data_columns:
        # col_count = len(train[col].unique())
        # config[col] = col_count
        config[str(col)] = dset.bin_number
    config["label"] = y_dim
    # print(config)

    if args.if_evaluate:
        ## load pre-trained model
        model = load_object(os.path.join(save_dir, "model.pkl"))
    else:
        ### Set up model and training
        if args.use_mpc:
            print("Using MPC-based Private PGM")
            model = MPC_PrivatePGM(
                "label",
                args.enable_privacy,
                args.target_epsilon,
                args.target_delta,
                mpc_path=args.mpc_path,
                protocol=args.mpc_protocol,
            )
        else:
            print("Using centralized Private PGM")
            model = Private_PGM(
                "label", args.enable_privacy, args.target_epsilon, args.target_delta
            )
        save_object(model, os.path.join(save_dir, "model.pkl"))
        model.train(train_dataframe, config, num_iters=args.num_iters)
        save_object(model, os.path.join(save_dir, "model.pkl"))

    ### Eval
    fake_data = []
    fake_label = []
    for i in range(args.num_samples_ratio):
        syn_data = model.generate()
        X_syn, y_syn = syn_data[:, :-1], syn_data[:, -1]
        fake_data.append(X_syn)
        fake_label.append(y_syn)
        fake_data_ = np.concatenate(fake_data)
        fake_label_ = np.concatenate(fake_label)
        print("-----fake data----")
        print(fake_data_[:10])
        print("----train data----")
        print(train_x[:10])
        save_data_csv(
            os.path.join(
                save_dir, f"samples/k{args.split_seed}_s{args.random_seed}.csv"
            ),
            dset._inverse_transform(fake_data_),
            fake_label_,
            dset.column_names,
        )
        who_2022_labels = [dset.label_map[int(l)] for l in fake_label_]

        save_data_csv(
            os.path.join(save_dir, f"samples/k{args.split_seed}_s{args.random_seed}_labeled.csv"),
            dset._inverse_transform(fake_data_),
            who_2022_labels, # Use the string labels
            dset.column_names,
            label_name="WHO_2022"
        )
        real_acc = Eval(fake_data_, fake_label_, model_type=eval_model_type).efficacy(
            test_x, test_y
        )
        fake_acc = Eval(test_x, test_y, model_type=eval_model_type).efficacy(
            fake_data_, fake_label_
        )

        fake_acc2 = Eval(
            dset._inverse_transform(test_x), test_y, model_type=eval_model_type
        ).efficacy(np.nan_to_num(dset._inverse_transform(fake_data_)), fake_label_)
        print("=" * 100)
        print("ratio= " + str(i))
        print("real acc: ", str(real_acc))
        print("fake acc: ", str(fake_acc))
        print("fake acc (original data space): ", str(fake_acc2))

        write_csv(
            os.path.join(save_dir, "eval.csv"),
            "ratio= " + str(i),
            [real_acc, fake_acc],
            ["real acc", "fake acc"],
        )

    return


if __name__ == "__main__":
    main()
