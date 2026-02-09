import random
import ipdb
import time
import torch
import torch.nn as nn
import torcheval.metrics
import numpy as np
import yaml
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import model.TSmodels as TSmodels
from sklearn.preprocessing import StandardScaler
import Augment
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_default_dtype(torch.float32)
    return None

def load_data_args(args):
    with open(args.config_filename) as f:
        config = yaml.full_load(f)
    args.device = 'cuda:'+ str(args.device) if torch.cuda.is_available() else 'cpu'
    data_args = config['data']
    args.num_classes = data_args[args.dataset]['num_classes']
    args.channel = data_args[args.dataset]['channel']
    args.time_step = data_args[args.dataset]['time_step']
    args.spl_search_paras = data_args[args.dataset]['spl_search_paras']
    aug_para = data_args[args.dataset]['aug_para']
    args.jitter_scale_ratio = aug_para['jitter_scale_ratio']
    args.jitter_ratio = aug_para['jitter_ratio']
    args.max_seg = aug_para['max_seg']
    return args


def load_data(args):
    args.classes = [str(i).zfill(5) for i in range(args.num_classes)]
    args.norm_info = None
    train_set = TsDataset("{}/{}/{}".format(args.data_path, args.dataset, 'train'), args, flag='train')
    # print(f'norm is {args.norm}, norm info is {args.norm_info}')
    val_set = TsDataset("{}/{}/{}".format(args.data_path, args.dataset, 'val'), args, flag='val') 
    test_set = TsDataset("{}/{}/{}".format(args.data_path, args.dataset, 'test'), args, flag='test') 
    train_loader = DataLoader(TensorDataset(train_set.data.to(args.device), train_set.target.to(args.device)), batch_size = args.train_bsz, shuffle = True)
    val_loader = DataLoader(TensorDataset(val_set.data.to(args.device), val_set.target.to(args.device)), batch_size = args.val_bsz, shuffle = True)
    test_loader = DataLoader(TensorDataset(test_set.data.to(args.device), test_set.target.to(args.device)), batch_size = args.val_bsz, shuffle = True)
    data_sets = {'train':train_set, 'val':val_set, 'test':test_set}
    data_loaders = {'train':train_loader, 'val': val_loader, 'test': test_loader}
    print(f"train:{train_set.data.shape}, val:{val_set.data.shape}, test:{test_set.data.shape}")
    return data_sets, data_loaders

def build_model(args, data_loaders, flag, specify_model=None):
    model_name = specify_model if specify_model else args.model
    print(f"buiding a {model_name}-{flag} model")
    # if args.scond == 1 and flag=='teacher':
    if args.scond == 1:
        model = TSmodels.get_network(args, model_name, 'sl').to(args.device)
    else:
        model = TSmodels.get_network(args, model_name).to(args.device)
    folder_name = Path("model_dict/")
    path = (folder_name/model_name/args.dataset/f"scond={args.scond}"/flag/args.model).with_suffix(".pth")
    if 'teacher' in flag and args.teacher_pretrain and path.exists() and not ('rand' in flag):
        print('using pretrained teacher')
    else:
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        print("Start training a {} model...\n--------------".format(flag))
        # train $flag model
        lr = args.lr_teacher if 'teacher' in flag and not 'rand' in flag else args.lr_stu
        print(f'training {flag}, lr: {lr}')
        # teacher_optim = torch.optim.SGD(model.parameters(), lr=lr)
        teacher_optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        teacher_optim.zero_grad()
        time0 = time.time()
        train_key = 'train' if 'teacher' in flag else 'train_syn'
        num_epochs = args.ttrain_epochs if 'teacher' in flag else args.strain_epochs
        lr_scheduler = lr_cosine_policy(lr, 0, num_epochs) # warm up = 10
        train_loss_type = 'CE' if 'teacher' in flag else 'KL'
        best_loss = 1e3
        best_metric = None
        best_e = None
        for e in range(num_epochs):
            lr_scheduler(teacher_optim, e, e)
            model.train()
            train_loss, train_metrics = one_epoch_train('train', data={'data_loader':data_loaders[train_key], 'model_name' : args.model}, net = model, optimizer=teacher_optim, loss_type=train_loss_type, args = args)
            with torch.no_grad():
                model.eval()
                test_loss, test_metrics = one_epoch_test('test', data={'data_loader':data_loaders['test'], 'model_name' : args.model}, net = model, loss_type='CE', args = args)
                if e % 10 == 0:
                   val_loss, val_metrics = one_epoch_test('val', data={'data_loader':data_loaders['val'], 'model_name' : args.model}, net = model, loss_type='CE', args = args) 
            train_str, test_str = 'Epoch: {}\tTrain Loss : {:.6f}\t'.format(e, train_loss), 'Epoch: {}\tTest Loss : {:.6f}\t'.format(e, test_loss)
            val_str = 'Epoch: {}\tVal Loss : {:.6f}\t'.format(e, val_loss)
            for k in train_metrics.keys():
                # print(k, train_metrics[k].compute())
                train_str += '{}: {:.4f}, '.format(k, train_metrics[k].compute())
                test_str += '{}: {:.4f}, '.format(k, test_metrics[k].compute())
                val_str += '{}: {:.4f}, '.format(k, val_metrics[k].compute())
            print(train_str)
            print(test_str)
            if e % 10 == 0:
                print(val_str)
            print("Cost : {:.3f}s".format(time.time()-time0))
            time0 = time.time()
            if best_loss > test_loss:
                best_loss = test_loss
                best_metric = test_metrics
                best_e = e
                torch.save(model.state_dict(), path)
        print("{} model best test metrics:{} at epoch {}".format(flag, [(k,v.compute().item()) for (k,v) in best_metric.items()], best_e))
        print("{} model saved to {}".format(flag, path))
    #----load best model----#
    model.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        model.eval()
        val_loss, val_metrics = one_epoch_test('val', data={'data_loader':data_loaders['val'], 'model_name' : args.model}, net=model, loss_type='CE', args = args)
        print("{} model val metrics: {}".format(flag, [(k,v.compute().item()) for (k,v) in val_metrics.items()]))
    return model, val_metrics['Accuracy'].compute().item()

def one_epoch_train(mode, data, net, optimizer, loss_type, args, texture=False):
    if loss_type == 'CE':
        criterion = nn.CrossEntropyLoss()
    elif loss_type == 'KL':
        criterion = nn.KLDivLoss(reduction='batchmean')
    else:
       raise TypeError('Loss Type for train must be CE or KL') 
    loss_avg, num_b = 0, 0
    net = net.to(args.device)
    net.train()
    data_loader,  model_name = data['data_loader'], data['model_name']
    metrics = {
        'Accuracy':torcheval.metrics.MulticlassAccuracy(),
        'Precision':torcheval.metrics.MulticlassPrecision(num_classes = args.num_classes, average = 'macro'),
        'Recall':torcheval.metrics.MulticlassRecall(num_classes = args.num_classes, average = 'macro'),
        'F1':torcheval.metrics.MulticlassF1Score(num_classes = args.num_classes, average = 'macro'),
        'AUROC':torcheval.metrics.MulticlassAUROC(num_classes = args.num_classes, average = 'macro'),
        'AUPRC':torcheval.metrics.MulticlassAUPRC(num_classes = args.num_classes, average = 'macro')
    }
    for met in metrics.values():
        met.to(args.device)
    inputaug_list = list(args.inputaug.split('_'))
    optimizer.zero_grad()
    for idx, (x, y) in enumerate(data_loader):
        x = x.to(args.device)
        y = y.to(torch.long).to(args.device)
        for inputaug in inputaug_list:
            x_aug = Input_Augmentation(x, inputaug, args)
            y_pred = net(x_aug)
            if loss_type == 'CE':
                loss = criterion(y_pred, y)
            elif loss_type == 'KL':
                y_pred_prob = torch.softmax(y_pred/args.temperature, dim=-1)
                y_soft_prob = torch.softmax(args.teacher(x_aug).detach()/args.temperature, dim=-1)
                loss = criterion(torch.log(y_pred_prob), y_soft_prob)
            else:
                raise TypeError('Loss Type is not CrossEntropyLoss') 
            loss_avg += loss.item()
            n_b = y.shape[0]
            num_b += n_b
            prob = F.softmax(y_pred, dim=1)
            for k, met in metrics.items():
                met.update(prob, y)
            loss = loss / len(inputaug_list)
            loss.backward()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        if (idx + 1) % args.gradient_accumulation_steps == 0 or idx == len(args.train_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()
    loss_avg = loss_avg / num_b
    return loss_avg, metrics

def one_epoch_test(mode, data, net, loss_type, args):
    if loss_type == 'CE':
        criterion = nn.CrossEntropyLoss()
    else:
        raise TypeError('Loss Type for test must be CrossEntropyLoss') 
    loss_avg, num_b = 0, 0
    net = net.to(args.device)
    net.eval()
    data_loader,  model_name = data['data_loader'], data['model_name']
    metrics = {
        'Accuracy':torcheval.metrics.MulticlassAccuracy(),
        'Precision':torcheval.metrics.MulticlassPrecision(num_classes = args.num_classes, average = 'macro'),
        'Recall':torcheval.metrics.MulticlassRecall(num_classes = args.num_classes, average = 'macro'),
        'F1':torcheval.metrics.MulticlassF1Score(num_classes = args.num_classes, average = 'macro'),
        'AUROC':torcheval.metrics.MulticlassAUROC(num_classes = args.num_classes, average = 'macro'),
        'AUPRC':torcheval.metrics.MulticlassAUPRC(num_classes = args.num_classes, average = 'macro')
    }
    for met in metrics.values():
        met.to(args.device)
    for idx, (x, y) in enumerate(data_loader):
        x = x.to(args.device)
        y = y.to(torch.long).to(args.device)
        with torch.no_grad():
            y_pred = net(x)
        loss = criterion(y_pred, y)
        n_b = y.shape[0]
        loss_avg += loss.item() * n_b
        prob = F.softmax(y_pred, dim=1)
        # print(prob)
        # print(MSE, RMSE, MAE, MAPE)
        for k, met in metrics.items():
            met.update(prob, y)
        num_b += n_b
    loss_avg /= num_b
    return loss_avg, metrics

# def one_epoch(mode, data, net, sl_net, optimizer, loss_type, args, texture=False):
    # if loss_type == 'CE':
        # criterion = nn.CrossEntropyLoss()
    # elif loss_type == 'KL':
        # criterion = nn.KLDivLoss(reduction='batchmean')
    # else:
       # raise TypeError('Loss Type is not CrossEntropyLoss') 
    # loss_avg, num_b = 0, 0
    # net = net.to(args.device)
    # data_loader,  model_name = data['data_loader'], data['model_name']
    # net.train() if 'train' in mode else net.eval()
    # metrics = {
        # 'Accuracy':torcheval.metrics.MulticlassAccuracy(),
        # 'Precision':torcheval.metrics.MulticlassPrecision(num_classes = args.num_classes, average = 'macro'),
        # 'Recall':torcheval.metrics.MulticlassRecall(num_classes = args.num_classes, average = 'macro'),
        # 'F1':torcheval.metrics.MulticlassF1Score(num_classes = args.num_classes, average = 'macro'),
        # 'AUROC':torcheval.metrics.MulticlassAUROC(num_classes = args.num_classes, average = 'macro'),
        # 'AUPRC':torcheval.metrics.MulticlassAUPRC(num_classes = args.num_classes, average = 'macro')
    # }
    # for met in metrics.values():
        # met.to(args.device)
    # inputaug_list = list(args.inputaug.split('_'))
    # optimizer.zero_grad()
    # for idx, (x, y) in enumerate(data_loader):
        # x = x.to(args.device)
        # y = y.to(torch.long).to(args.device)
        # for inputaug in inputaug_list:
            # if 'train' in mode:
                # x_aug = Input_Augmentation(x, inputaug, args)
                # y_pred = net(x_aug)
            # else:
                # with torch.no_grad():
                    # y_pred = net(x)
            # if loss_type == 'CE':
                # loss = criterion(y_pred, y)
            # elif loss_type == 'KL':
                # y_pred_prob = torch.softmax(y_pred, dim=-1)
                # with torch.no_grad():
                    # y_soft = args.teacher(x_aug)
                    # y_soft_prob = torch.softmax(y_soft, dim=-1)
                # loss = criterion(torch.log(y_pred_prob), y_soft_prob)
            # else:
                # raise TypeError('Loss Type is not CrossEntropyLoss') 
            # n_b = y.shape[0]
            # loss_avg += loss.item() * n_b
            # prob = F.softmax(y_pred, dim=1)
            # # print(prob)
            # # print(MSE, RMSE, MAE, MAPE)
            # for k, met in metrics.items():
                # met.update(prob, y)
            # num_b += n_b
            # if('train' in mode):
                # loss = loss / len(inputaug)
                # if args.gradient_accumulation_steps > 1:
                    # loss = loss / args.gradient_accumulation_steps
                # loss = loss / len(inputaug_list)
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
    # loss_avg /= num_b
    # return loss_avg, metrics

def Input_Augmentation(X, inputaug, args):
    if (inputaug == 'raw'):
        return X
    if (inputaug == 'weak1'):
        return Augment.scaling(X, args)
    if (inputaug == 'weak2'):
        return Augment.jitter(X, args)
    if (inputaug == 'strong'):
        return Augment.jitter(Augment.scaling(X, args), args)
    if (inputaug == 'slweak1'):
        return Augment.sl_invariant_scale(X, args)
    if (inputaug == 'slweak2'):
        return Augment.sl_invariant_jitter(X, args)
    if (inputaug == 'slstrong'):
        return Augment.sl_invariant_jitter(Augment.sl_invariant_scale(X, args), args)

def get_images_clustering(c, n):
        all_imgaes = torch.tensor(data_devided['train']['samples'][indices_class[c]]).to(torch.float).to(args.device)
        temp_model = args.model
        args.model = 'CNNBN'
        net_temp = TSmodels.get_network(args).to(args.device)
        args.model = temp_model
        print(all_imgaes.shape)
        embeddings = net_temp.embed(all_imgaes)
        kmeans = KMeans(n_clusters=args.ipc, mode='euclidean', verbose=1)
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.centroids
        
        dis_mat_torch = torch.cdist(centers,embeddings,p=2)
        clustered_images = all_imgaes[torch.argmin(dis_mat_torch,dim=1)]
        return clustered_images

class TsDataset():
    def __init__(self, data_folder, args, flag, specify_ipc=None):
        self.data = []
        self.target = []
        for c in range(args.num_classes):
            file_path = data_folder + "/" + args.classes[c] + '.npy'
            x = torch.FloatTensor(np.load(file_path))
            y = torch.IntTensor([c] * x.shape[0])
            if args.mipc or specify_ipc:
                if specify_ipc:
                    self.data.append(x[: specify_ipc])
                    self.target.append(y[: specify_ipc])
                else: # args.mipc
                    idxs = torch.randperm(x.shape[0])[: args.mipc]
                    self.data.append(x[idxs])
                    self.target.append(y[idxs])
            else:
                self.data.append(x)
                self.target.append(y)
        self.data = torch.cat(self.data)
        self.target = torch.cat(self.target)
        if flag == 'train':
            # check for class imbalance
            class_distribution = Counter(self.target.numpy())
            min_class_size = min(class_distribution.values())
            max_class_size = max(class_distribution.values())
            imbalance_ratio = min_class_size / max_class_size
            imbalance_threshold = args.imbalance_ratio
            # Flag to indicate whether resampling was done
            resampling_done = False
            # Initialize resampled data with original data
            data_resampled, target_resampled = self.data, self.target
            if imbalance_ratio < imbalance_threshold:
                print("Class imbalance detected. Applying RandomOverSampler...")
                ros = RandomOverSampler(random_state=0)
                sample_num, channels, features = self.data.shape
                self.data = self.data.reshape(sample_num, -1)
                self.data, self.target = ros.fit_resample(self.data, self.target)
                resampling_done = True
                self.data = self.data.reshape(self.data.shape[0], channels, features)
                self.data = torch.from_numpy(self.data)
                self.target = torch.from_numpy(self.target)
                print(f"resampled train data. imbalance_ratio is {imbalance_ratio}")
        if args.norm:
            if args.norm_info == None:
                args.norm_info = {}
                args.norm_info['mean'], args.norm_info['std'] = [stat.unsqueeze(0).unsqueeze(2) for stat in self.mean_std2(self.data)]
            self.data = self.mean_std_transform(args.norm_info['mean'], args.norm_info['std'])
            
    def mean_std_transform(self, mean, std):
        return (self.data - mean) / std
    
    def mean_std1(self, data):
        m_len = torch.mean(data, axis=2)
        mean = torch.mean(m_len, axis=0)

        s_len = torch.std(data, axis=2)
        std = torch.max(s_len, axis=0)
        return mean, std
    
    def mean_std2(self, data):
        mean = torch.mean(data, axis=(0, 2)) 
        std = torch.std(data, axis=(0, 2))
        return mean, std

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.target)
    

class BNFeatureHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        input = input[0]
        if len(input.shape) == 2:
            input = input.unsqueeze(2) # 改成 （N, C, 1）
        nch = input.shape[1]
        mean = input.mean([0, 2])
        var = input.permute(1, 0, 2).contiguous().reshape([nch, -1]).var(1, unbiased=False)
        r_feature = torch.norm(module.running_var.data - var, p=2) + torch.norm(module.running_mean.data - mean, p=2)
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()

def get_continuous_losses(inputs):
    diff = inputs[:, :, :-1] - inputs[:, :, 1:]
    loss_var_l2 = torch.norm(diff) 
    return loss_var_l2

def cal_dist(args, x_base, x):
    dist_mat = []
    for si in args.shapelets_info:
        seg = x_base[int(si[0]), int(si[5]), int(si[1]):int(si[2])]
        st = max(int(si[1]) - args.window_size, 0)
        ed = st + int(si[2]) - int(si[1])
        min_dist = torch.ones(x.shape[0]) * 100
        while ed <= args.time_step and ed <= si[2] + args.window_size:
            min_dist = torch.min(torch.norm(x[:, int(si[5]), st: ed] - seg, dim=-1)/(torch.norm(x[:, int(si[5]), st: ed] + torch.norm(seg))), min_dist)
            st += 2
            ed += 2
        dist_mat.append(min_dist)
    dist_mat = torch.stack(dist_mat, dim=-1)
    return dist_mat

from pyts.transformation import ShapeletTransform
import pathlib
import pickle

class ShapeletPyts():
    def __init__(self, args, n_shapelets='auto', criterion='mutual_info',
                 window_sizes='auto', window_steps=None,
                 remove_similar=True, sort=True, verbose=0, random_state=None,
                 n_jobs=None):
        super().__init__()
        self.n_shapelets = n_shapelets
        self.criterion = criterion
        self.window_sizes = window_sizes
        self.window_steps = window_steps
        self.remove_similar = remove_similar
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.path = (pathlib.Path(args.sc_path) / args.dataset).with_suffix('.pkl')
        self.sort = True
        self.n_channels = args.channel
        self.models = [ShapeletTransform(n_shapelets=self.n_shapelets, criterion=self.criterion,
                 window_sizes=self.window_sizes, window_steps=self.window_steps, 
                 fit_steps=args.fit_steps, transform_steps=args.transform_steps,
                 obs_window_size=args.obs_window_size,
                 remove_similar=self.remove_similar, 
                 sort=self.sort, verbose=self.verbose, random_state=self.random_state, n_jobs=self.n_jobs, fast=args.fast) for i in range(self.n_channels)]
        self.load = False
        self.args = args
    
    def load_shapelets_info(self, n_shapelets, path):
        self.load = True
        with open(path, 'rb') as f:
            self.loaded_info = pickle.load(f)
        num_require = n_shapelets 
        num_saved = self.loaded_info[0]['shapelets_'].shape[0]
        if num_saved < num_require:
            raise AssertionError("requiring shapelets more than saved amount. You can regenerate shapelets by --pre_shapelet_discovery 0.")
        for i in range(self.n_channels):
            self.models[i].shapelets_ = self.loaded_info[i]['shapelets_'][:n_shapelets]
            self.models[i].indices_ = self.loaded_info[i]['indices_'][:n_shapelets]
            self.models[i].c_ = self.loaded_info[i]['c_'][:n_shapelets]
            self.models[i].scores_ = self.loaded_info[i]['scores_'][:n_shapelets]
        print(f'from {path} load shaplets sucess!')
    
    def return_shapelets_info(self):
        def group_by(shapelets, indices):
            lengths = np.array([len(shapelet) for shapelet in shapelets])
            uniqs = list(set(lengths))
            idxs = np.array([i for i in range(len(shapelets))])
            grouped_idxs = [idxs[lengths == len_i] for len_i in uniqs]
            grouped_shapelets = [torch.from_numpy(np.stack(shapelets[idxs_len_i], axis=0)).to(self.args.device, dtype=torch.float32) for idxs_len_i in grouped_idxs]
            grouped_indices = [torch.from_numpy(np.stack(indices[idxs_len_i], axis=0)).to(self.args.device, dtype=torch.float32) for idxs_len_i in grouped_idxs]
            return grouped_shapelets, grouped_indices
        grouped_shapelets = [group_by(self.models[i].shapelets_, self.models[i].indices_)[0] for i in range(self.n_channels)]
        grouped_indices = [group_by(self.models[i].shapelets_, self.models[i].indices_)[1] for i in range(self.n_channels)]
        return grouped_shapelets, grouped_indices

    def fit_transform_save(self, X, y, path):
        save_dict = {}
        ret = np.zeros((X.shape[0], self.n_channels, self.n_shapelets))
        for i in range(self.n_channels):
            ret[:, i, :] = self.models[i].fit_transform(X[:, i, :], y)
            self.models[i].c_ = y[self.models[i].indices_[:,0]]
            save_dict[i] = {'shapelets_': self.models[i].shapelets_, 'indices_': self.models[i].indices_, 'c_': self.models[i].c_, 'scores_': self.models[i].scores_}
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        return ret
    
    def transform(self, X):
        is_tensor=False
        device=None
        if torch.is_tensor(X):
            is_tensor=True
            device=X.device
            if X.requires_grad:
                X = X.detach()
            X = X.cpu().numpy()
        # ret = np.zeros((X.shape[0], len_once * self.n_channels))
        ret = []
        if self.load:
            for i in range(self.n_channels):
                # ret[:, len_once * i : len_once * (i+1)] = self.models[i]._transform(X[:, i, :])
                ret.append(self.models[i]._transform(X[:, i, :]))
        else:
            for i in range(self.n_channels):
                # ret[:, len_once * i : len_once * (i+1)] = self.models[i].transform(X[:, i, :])
                ret.append(self.models[i].transform(X[:, i, :]))
        if not is_tensor:
            return np.stack(ret, 1)
        return torch.from_numpy(np.stack(ret, 1)).to(device=device, dtype=torch.float32)
    
    def _top_ipc2(self, labels, scores, ipc): # 不分channel
        label_to_indices = {label: [] for label in range(self.args.num_classes)}
        # 遍历 labels，记录每个标签对应的样本索引
        lab_scr =list(enumerate(zip(labels, scores)))
        lab_scr.sort(reverse=True, key=lambda x: x[1][1])
        for idx, (label, score) in lab_scr:
            if len(label_to_indices[label]) < ipc:  # 只取前两个样本的索引
                label_to_indices[label].append(idx)
        ret_idxs = np.array([idx for label in range(self.args.num_classes) for idx in label_to_indices[label]])
        return ret_idxs

class differentiable_shapelets():
    def __init__(self, shapelets, indices):
        super().__init__()
        self.shapelets = shapelets # on gpu. grouped by lengths
        self.indices = indices # on cpu 
    
    def transform(self, X, transform_steps=1):
        channels = len(self.shapelets)
        assert channels == X.shape[1]
        ret = []
        for i in range(channels):
            # shapelets, infos for each channel
            shapelets = self.shapelets[i]
            indices = self.indices[i]
            for len_i_shapelets, len_i_infos in zip(shapelets, indices):
                # shapelets, infos for each length
                X_view = X[:, i, :] # (bsz, time_steps)
                windows = X_view.unfold(dimension=1, size=len_i_shapelets.shape[1], step=transform_steps) # (bsz, num_windows, length)
                bsz, num_windows, length = windows.shape
                dists = torch.cdist(windows.reshape(-1, length).unsqueeze(1), len_i_shapelets.unsqueeze(0), p=2) + 1e-6 # (bsz * num_windows, num_shapelets)
                dists = dists.reshape(bsz, num_windows, -1)
                min_dist = dists.min(dim=1)[0] # (bsz, num_shapelets)
                ret.append(min_dist)
        ret = torch.cat(ret, dim=1)
        return ret

        
def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn) 

def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr

        
def random_pairs_of_indices(target, k):
    c = torch.unique(target).tolist()
    c.sort()
    indices = [torch.nonzero(target == i).flatten().tolist() for i in c]
    pairs = []
    for i in range(len(c)):
        pairs += random.sample(indices[i], k)
    pairs = torch.tensor(pairs).reshape(-1, k).transpose(0, 1)
    return pairs        

def cal_mask(num_channels, num_points, args):
    indices_list = [torch.cat(cha, axis=0)[:, 1:3].to(torch.long) for cha in args.sl_net.indices]
    mask = torch.zeros(num_channels, num_points)
    for channel_i in range(num_channels):
        indices = indices_list[channel_i]
        mask_channel_i = torch.zeros(num_points+1, dtype=torch.long)
        mask_channel_i[indices[:, 0]] = 1
        mask_channel_i[indices[:, 1]] = -1
        mask_channel_i = mask_channel_i[:num_points]
        record = 0
        for i in range(len(mask_channel_i)):
            mask_channel_i[i] += record
            record = mask_channel_i[i]
        mask_channel_i = mask_channel_i > 0
        mask[channel_i, mask_channel_i] = 1
    return mask.to(device=args.device, dtype=torch.int32)
    