import os
import argparse
import torch
import torch.nn as nn
import tqdm
import numpy as np
from utils import *
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import time
import pathlib


def main(args):
    all_acc = []
    for random_state in range(2021, 2026):
        args.random_state = random_state
        data_sets, data_loaders = load_data(args)
        print(f"----------imbalance ratio: {args.imbalance_ratio}-----------")
        print("-------------data loaded!-------------")
        
        # --------------------------------------------------------------------------------------------------------------
        # -------------------------------------------Shapelet Discovery Stage-------------------------------------------
        # generally, we set max_length of searching as around 1/3 length of time series length.
        st_idx, ed_idx, size_step, win_step = args.spl_search_paras
        print(f"train data shape: {data_sets['train'].data.shape}")
        window_sizes = np.arange(100, 1000, 100)
        window_steps = np.array([100]* len(window_sizes))

        shapelet_model = ShapeletPyts(args=args, window_sizes=window_sizes, 
                            n_shapelets=args.num_shapelets, random_state=args.random_state, 
                            n_jobs=args.num_processes, window_steps=window_steps)
        shapelet_path = (pathlib.Path(args.sc_path) / args.dataset).with_suffix(".pkl")
        if args.pre_shapelet_discovery and shapelet_path.is_file():
            shapelet_model.load_shapelets_info(n_shapelets=args.num_shapelets, path=shapelet_path) # num_shapelets < 储存的shapelets数量
            print("use presaved shapelets")
            feats_train = shapelet_model.transform(data_sets["train"].data.numpy())
        else:
            time_s = time.time()
            if shapelet_path.is_file():
                raise FileExistsError(f"shapelet file exist at {shapelet_path}. check if use --pre_shapelet_discovery 1. ")
            feats_train = shapelet_model.fit_transform_save(X=data_sets["train"].data, y=data_sets["train"].target, path=shapelet_path)
            print(f"shapelet discovery time: {(time.time() - time_s)/60:.2f} min")
        grouped_shapelets, grouped_indices = shapelet_model.return_shapelets_info()
        args.sl_net = differentiable_shapelets(grouped_shapelets, grouped_indices)
        # args.sl_net.transform(data_sets["train"].data[:10].to(args.device))
        # 计算mask
        _n, num_channels, num_points = data_sets["train"].data.shape
        args.mask = cal_mask(num_channels, num_points, args)

        # ----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------classify directly by shapelet--------------------------------------------
        feats_train = []
        feats_train_y = []
        for idx, (x_train, y_train) in enumerate(data_loaders["train"]):
            x_train = x_train.to(args.device)
            feats_train.append(args.sl_net.transform(x_train).to('cpu'))
            feats_train_y.append(y_train)
        feats_train = torch.cat(feats_train, dim=0)
        feats_train = feats_train.reshape(feats_train.shape[0], -1)
        feats_train_y = torch.cat(feats_train_y, dim=0).to('cpu')

        feats_test = []
        feats_test_y = []
        for idx, (x_test, y_test) in enumerate(data_loaders["test"]):
            x_test = x_test.to(args.device)
            feats_test.append(args.sl_net.transform(x_test).to('cpu'))
            feats_test_y.append(y_test)
        feats_test = torch.cat(feats_test, dim=0)
        feats_test = feats_test.reshape(feats_test.shape[0], -1)
        feats_test_y = torch.cat(feats_test_y, dim=0).to('cpu')
            
        clf = LogisticRegression(max_iter=8000)
        clf.fit(feats_train, feats_train_y)
        score_test = clf.score(feats_test, feats_test_y)
        print(f"feat_train: {feats_train.shape}")
        print(f"test acc is {score_test:.4f}")

        # --------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------Knowledge Featching Stage---------------------------------------
        teacher, _acc = build_model(args, data_loaders, flag="teacher")
        print(data_sets["train"].data.shape)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        args.teacher = teacher

        # --------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------Data Synthesis Stage--------------------------------------------
        if args.scond in [0, 1]:
            # intial inputs
            init_inputs_index = random_pairs_of_indices(data_sets["train"].target, k=args.ipc)
            all_inputs = []
            all_targets = []
            for i in range(args.ipc):
                print("synthesizing ipc idx = {}".format(i))
                targets = torch.LongTensor(np.arange(args.num_classes)).to(args.device)
                # inputs = torch.randn((targets.shape[0], args.channel, args.time_step), requires_grad=True, device=args.device, dtype=torch.float)
                inputs = data_sets["train"].data[init_inputs_index[i]].to(args.device)
                inputs.requires_grad = True
                loss_r_feature_layers = []
                for module in teacher.features:
                    if isinstance(module, nn.BatchNorm1d):
                        loss_r_feature_layers.append(BNFeatureHook(module))
                criterion = nn.CrossEntropyLoss()
                optim_r = torch.optim.Adam([inputs], lr=args.lr_r, betas=[0.5, 0.9])
                lr_scheduler = lr_cosine_policy(args.lr_r, 0, args.re_epochs) # warm up length=0
                best_loss = 1e3
                best_inputs = None
                best_epoch = 0
                inputaug_list = list(args.inputaug.split("_"))
                optim_r.zero_grad()
                for e in range(args.re_epochs):
                    lr_scheduler(optim_r, e, e)
                    loss = 0
                    # We find the impact of augmentation varies. It does not always improve the performance.
                    for inputaug in inputaug_list:
                        inputs_aug = Input_Augmentation(inputs, inputaug, args)
                        outputs = teacher(inputs_aug)
                        # 1
                        loss_ce = criterion(outputs, targets)
                        loss_ce = loss_ce * 0.99
                        # 2
                        rescale = [args.first_bn_multiplier] + [1. for _ in range(len(loss_r_feature_layers)-2)] + [args.last_bn_multiplier]
                        loss_r_bn_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])
                        loss_r_bn_feature = loss_r_bn_feature * 0.01 # range [0.005, 0.01] generally produce acceptable results.
                        # 3
                        loss_l2 = torch.norm(inputs_aug.reshape(inputs_aug.shape[0], -1), dim=1).mean()
                        # loss_l2 = loss_l2 * 0.001
                        loss_l2 = loss_l2 * 0.00
                        # 4
                        loss_var_l2 = get_continuous_losses(inputs_aug)
                        # loss_var_l2 = loss_var_l2 * 0.001
                        loss_var_l2 = loss_var_l2 * 0.00
                        # 5 
                        # loss_l2 and loss_var_l2 seems not applied to times series.
                        loss_aug = loss_ce + loss_r_bn_feature + loss_l2 + loss_var_l2
                        loss_aug = loss_aug / len(inputaug_list)
                        loss_aug.backward()
                        loss += loss_aug.item()
                        # 6
                    if best_loss > loss or e == 0:
                        best_loss = loss
                        best_inputs = inputs.data.clone()
                        best_epoch = e
                    if e % args.interval == 0 or e == args.re_epochs - 1:
                        print(f"loss: {loss:.4f} = ce: {loss_ce:.4f} + bn: {loss_r_bn_feature:.4f} + l2: {loss_l2:.4f} + varl2: {loss_var_l2:.4f}")
                    optim_r.step()
                    optim_r.zero_grad()
                all_inputs.append(best_inputs)
                all_targets.append(targets)
                print(f"saving best inputs with loss: {best_loss} at epoch {best_epoch}")
            all_inputs = torch.cat(all_inputs, dim=0)
            if args.norm:
                all_inputs = (all_inputs * args.norm_info["std"].to(args.device)) + args.norm_info["mean"].to(args.device)
            all_targets = torch.cat(all_targets, dim=0)
            syn_save_dir = args.syn_data_path + "/" + args.model + "/" + args.dataset + "/" + "scond=" + str(args.scond)
            if not os.path.exists(syn_save_dir):
                os.makedirs(syn_save_dir)
            for i in range(args.num_classes):
                ci = args.classes[i]
                data_ci = all_inputs[all_targets == i]
                np.save(syn_save_dir + "/" + ci + ".npy", data_ci.detach().cpu().numpy())
            print("Syn data complished, saved to dir: {}\n{}".format(args.syn_data_path + "/" + args.dataset + "scond=" + str(args.scond), "-"*50))

        # --------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------validate syn data-----------------------------------------------

            syn_set = TsDataset("{}/{}/{}/{}".format(args.syn_data_path, args.model, args.dataset, "scond=" + str(args.scond)), args, "train", specify_ipc=args.ipc)
            syn_train_l = DataLoader(TensorDataset(syn_set.data.to(args.device), syn_set.target.to(args.device)), batch_size = args.val_bsz, shuffle = True)
            syn_pred = torch.argmax(teacher(syn_set.data.to(args.device)), dim=-1)
            print("expect labels:{}, predict syn_set labels:{}".format(syn_set.target, syn_pred))
            data_loaders["train_syn"] = syn_train_l
            student, acc = build_model(args, data_loaders, flag="stu")
            all_acc.append(acc)
        
        # --------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------prove shapelet knowledge-----------------------------------------------

            syn_set = TsDataset("{}/{}/{}/{}".format(args.syn_data_path, args.model, args.dataset, "scond=" + str(args.scond)), args, "train", specify_ipc=args.ipc)
            syn_feat = args.sl_net.transform(syn_set.data.to(args.device, torch.float32)).to('cpu')
            syn_feat = syn_feat.reshape(syn_feat.shape[0], -1)
            score = clf.score(syn_feat, syn_set.target)
            print(f"scond={args.scond}, classified by shapelet clf, score on syn data: {score:.4f}")

        # --------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------validate random data-----------------------------------------------
        else: # scond = -1: rand_teacher
            rand_set = TsDataset("{}/{}/{}".format(args.data_path, args.dataset, "train"), args, "train", specify_ipc=args.ipc)
            rand_train_l = DataLoader(TensorDataset(rand_set.data.to(args.device), rand_set.target.to(args.device)), batch_size = args.val_bsz, shuffle = True)
            teacher_pred_label = torch.argmax(teacher(rand_set.data.to(args.device)), dim=-1)
            print(f"real label: {rand_set.target}, pred by teacher: {teacher_pred_label}")
            data_loaders["train"] = rand_train_l
            teacher_rand, acc = build_model(args, data_loaders, flag="teacher_rand")
            all_acc.append(acc)
    all_acc = np.array(all_acc)
    print(f"Accuracy: {np.mean(all_acc):.4f} +- {np.std(all_acc):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Processing")
    parser.add_argument("--config_filename", default="config.yml")
    parser.add_argument("--random_state", type=int, default=2025, help="random state")
    parser.add_argument("--dataset", type=str, default="har", help="dataset")
    parser.add_argument("--data_path", type=str, default="data", help="dataset path")
    parser.add_argument("--sc_path", type=str, default="sc", help="shapelet candidates path")
    parser.add_argument("--device", type=int, default=3, help="specify a gpu")
    parser.add_argument("--num_processes", type=int, default=70, help="multiprocess.")
    parser.add_argument("--mipc", type=int, default=None, help="number of pre-loaded images per class" )
    parser.add_argument("--ipc", type=int, default=1, help="image(s) per class")
    parser.add_argument("--train_bsz", type=int, default=256, help="number of train images per class" )
    parser.add_argument("--val_bsz", type=int, default=20, help="number of val images per class" )
    parser.add_argument("--pre_shapelet_discovery", default=1, type=int, help="1 for pretrian, 0 for train")
    parser.add_argument("--teacher_pretrain", default=1, type=int, help="1 for pretrian, 0 for train")
    parser.add_argument("--model", type=str, default="CNNBN", help="model")
    parser.add_argument("--lr_teacher", type=float, default=0.005, help="learning rate for updating network parameters")
    parser.add_argument("--lr_r", type=float, default=0.02, help="learning rate for updating network parameters in recover stage")
    parser.add_argument("--lr_stu", type=float, default=0.0005, help="learning rate for updating network parameters in student model")
    parser.add_argument("--temperature", type=float, default=4, help="mix for 4, cutmix for 20")
    parser.add_argument("--momentum", type=float, default=0.9, help="sgd momentum")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="sgd weight decay")
    parser.add_argument("--ttrain_epochs", type=int, default=50)
    parser.add_argument("--re_epochs", type=int, default=50)
    parser.add_argument("--strain_epochs", type=int, default=100)
    parser.add_argument("--interval", type=int, default=50, help="save interval")
    parser.add_argument("--num_shapelets", default=1, type=int, help="number of shapelets")
    parser.add_argument("--first-bn-multiplier", type=float, default=10.,
                    help="additional multiplier on first bn layer of R_bn")
    parser.add_argument("--last-bn-multiplier", type=float, default=1.,
                    help="additional multiplier on sl bn layer")
    parser.add_argument("--syn_data_path", type=str, default="syn_data", help="dataset path")
    parser.add_argument("--inputaug", type=str, default="raw", help="use which augmentations, raw_weak_strong.")
    parser.add_argument("--norm", type=int, default=1, help="1 for norm, 0 for no norm")
    parser.add_argument("--scond", type=int, default=1, help="1 for shapecond, 0 for sre2l, - for rand_teacher")
    parser.add_argument("--gradient-accumulation-steps", type=int,
                        default=1, help="gradient accumulation steps for small gpu memory")
    parser.add_argument("--fit_steps", type=int, default=1, help="shapelet fit interval")
    parser.add_argument("--transform_steps", type=int, default=1, help="shapelet transform interval")
    parser.add_argument("--obs_window_size", type=int, default=10, help="observed window size")
    parser.add_argument("--imbalance_ratio", type=float, default=1.0, help="imbalance ratio")
    parser.add_argument("--fast", type=int, default=0, help="if 1, use fast shapelet version")
    args = parser.parse_args()
    args = load_data_args(args)


    set_seed(args.random_state)

    print(args)
    print("-----------------")
    main(args)
