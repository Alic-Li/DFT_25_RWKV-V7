import copy
import logging
import os
import pickle
import datetime
import numpy as np
import pandas as pd
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.tensorboard import SummaryWriter
from DFT import DFT
from my_lr_scheduler import ChainedScheduler
import random
os.environ['CUDA_VISIBLE_DEVICES'] = "6"  # 程序可见的GPU
cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


def set_random_seed(seed):
    # 设置Python随机种子
    random.seed(seed)

    # 设置NumPy随机种子
    np.random.seed(seed)

    # 设置PyTorch随机种子
    torch.manual_seed(seed)

    # 如果使用GPU，设置CUDA随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # 关闭cuDNN的自动优化，保证每次运行结果一致


class TRAINFIG:
    def __init__(self, model_name="DFT", GPU=0, universe='csi800', seed=11032,
                 dataset_country="CN_DATA",dataset="_2020_2023_step_5_c1c3",
                 n_epoch=75, lr=3e-4, gamma=1.0, coef=1.0,
                 cosine_period=4, T_0=15, T_mult=1, warmUp_epoch=10, eta_min=2e-5,
                 weight_decay=0.001, seq_len=8, d_feat=158, d_model=256, n_head=4, dropout=0.5,
                 gate_input_start_index=158, gate_input_end_index=221, beta=10,
                 train_stop_loss_threshold=0.95):
        self.model_name = model_name+f"{seed}_{universe}{dataset}"
        self.GPU = GPU
        self.universe = universe
        self.seed = seed
        self.dataset_dir_path = f"./DATASETS/{dataset_country}/{universe}"
        self.model_save_path = f"./model_params/{self.universe}/{self.model_name}"
        self.metrics_loss_path = f"./metrics/{self.universe}/{self.model_name}"
        self.log_dir = f"./logs/{self.model_name}"
        # 确保目录存在
        for path in [self.log_dir, self.model_save_path, self.metrics_loss_path]:
            os.makedirs(path, exist_ok=True)

        logging.basicConfig(filename=os.path.join(self.log_dir, f"{self.model_name}.log"),
                            level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(f"Train {self.model_name}")

        self.writer = SummaryWriter(log_dir=self.log_dir, filename_suffix=f"{self.model_name}")

        # 其他配置参数
        self.n_epoch = n_epoch
        self.lr = lr
        self.gamma = gamma
        self.coef = coef
        self.cosine_period = cosine_period
        self.T_0 = T_0
        self.T_mult = T_mult
        self.warmUp_epoch = warmUp_epoch
        self.eta_min = eta_min
        self.weight_decay = weight_decay
        parts = dataset.split('_')
        filtered_parts = list(filter(None, parts))
        self.seq_len = 8
        self.d_feat = d_feat
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = dropout
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.beta = beta
        self.train_stop_loss_threshold = train_stop_loss_threshold

        # 这些初始化应该在模型训练开始的地方进行
        self.device = torch.device(f"cuda:{self.GPU}" if torch.cuda.is_available() else "cpu")
        self.model = DFT(d_model=self.d_model, d_feat=self.d_feat, seq_len=self.seq_len,
                           t_nhead=self.n_head, S_dropout_rate=self.dropout, beta=self.beta).to(self.device)
        self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999),
                                          weight_decay=self.weight_decay)
        self.lr_scheduler = ChainedScheduler(self.train_optimizer, T_0=self.T_0, T_mul=self.T_mult,
                                             eta_min=self.eta_min, last_epoch=-1, max_lr=self.lr,
                                             warmup_steps=self.warmUp_epoch, gamma=self.gamma,
                                             coef=self.coef, step_size=3, cosine_period=self.cosine_period)

        # 日志记录也应根据实际操作动态添加，而非在此处静态定义
        self.writer.add_text("train_optimizer", self.train_optimizer.__str__())
        self.writer.add_text("lr_scheduler", self.lr_scheduler.__str__())

        self.logger.info(msg=f"\n===== Model {model_name} =====\n"
                             f"n_epochs: {n_epoch}\n"
                             f"start_lr: {lr}\n"
                             f"T_0: {T_0}\n"
                             f"T_mult: {T_mult}\n"
                             f"gamma: {gamma}\n"
                             f"coef: {coef}\n"
                             f"cosine_period: {cosine_period}\n"
                             f"eta_min: {eta_min}\n"
                             f"seed: {seed}\n"
                             f"optimizer: {self.train_optimizer}\n"
                             f"lr_scheduler: {self.lr_scheduler}\n"
                             f"description: train {model_name}\n\n")

        self.writer.add_text("model_name", model_name)
        self.writer.add_text("seed", str(seed))
        self.writer.add_text("n_head", str(n_head))
        self.writer.add_text("learning rate", str(lr))
        self.writer.add_text("T_0", str(T_0))
        self.writer.add_text("T_mult", str(T_mult))
        self.writer.add_text("gamma", str(gamma))
        self.writer.add_text("coef", str(coef))
        self.writer.add_text("eta_min", str(eta_min))
        self.writer.add_text("weight_decay", str(weight_decay))
        self.writer.add_text("cosine_period", str(cosine_period))


def calc_ic(pred, label):
    df = pd.DataFrame({'pred': pred, 'label': label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric


class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        super().__init__(data_source)
        self.data_source = data_source
        self.shuffle = shuffle
        # calculate number of samples in each batch
        self.daily_count = pd.Series(index=self.data_source.get_index(), dtype=np.float64).groupby(
            "datetime").size().values
        # calculate begin index of each batch
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


def loss_fn(pred, label):
    mask = ~torch.isnan(label)
    loss = (pred[mask] - label[mask]) ** 2
    return torch.mean(loss)


def _init_data_loader(data, shuffle=True, drop_last=True):
    sampler = DailyBatchSamplerRandom(data, shuffle)
    data_loader = DataLoader(
        data, sampler=sampler, drop_last=drop_last, num_workers=4, pin_memory=True)
    return data_loader


def train_epoch(data_loader, train_optimizer, lr_scheduler, model, device):
    model.train()
    losses = []

    for data in data_loader:
        data = torch.squeeze(data, dim=0)
        '''
        data.shape: (N, T, F)
        N - number of stocks
        T - length of lookback_window, 8
        F - 158 factors + 63 market information + 1 label           
        '''
        feature = data[:, :, 0:-1].to(device)
        label = data[:, -1, -1].to(device)
        # print("feature",feature.shape)
        # print("label",label.shape)
        # a=1/0

        pred = model(feature.float())

        loss = loss_fn(pred, label)
        losses.append(loss.item())

        train_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
        train_optimizer.step()
    lr_scheduler.step()

    return float(np.mean(losses))


def valid_epoch(data_loader, model, device):
    model.eval()
    losses = []
    ic = []
    ric = []
    with torch.no_grad():
        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(device)
            label = data[:, -1, -1].to(device)
            with torch.no_grad():
                pred = model(feature.float())
            loss = loss_fn(pred, label)
            losses.append(loss.item())

            daily_ic, daily_ric = calc_ic(
                pred.detach().cpu().numpy(), label.detach().cpu().numpy())
            ic.append(daily_ic)
            ric.append(daily_ric)

    metrics = {
        'IC': np.mean(ic),
        'ICIR': np.mean(ic) / np.std(ic),
        'RIC': np.mean(ric),
        'RICIR': np.mean(ic) / np.std(ric)
    }

    return float(np.mean(losses)), metrics


def train(model_name,dataset,universe,country,seed):
    TYPENAME = dataset
    # 实例化配置类
    TrainConfig = TRAINFIG(model_name=model_name,universe=universe, seed=seed,dataset_country=country,dataset=dataset)  # 自定义model_name

    # 获取当前时间
    current_time = datetime.datetime.now()
    # 生成文件名，格式为 模型名字_年月日时分秒.txt
    file_name = f"./txt/{TrainConfig.model_name}_{current_time.strftime('%Y%m%d%H%M%S')}.txt"
    with open(file_name, "w") as ftxt:
        ftxt.write("model_name=" + TrainConfig.model_name + "  universe=" + TrainConfig.universe + " seed=" + str(
            TrainConfig.seed) + "\n")

        if not os.path.exists(TrainConfig.dataset_dir_path):
            print(TrainConfig.dataset_dir_path)
            raise FileExistsError("Data dir not exists")

        universe = TrainConfig.universe  # or 'csi800'

        # Please install qlib first before load the data.
        data_name = ".pkl"
        with open(f'{TrainConfig.dataset_dir_path}//{universe}_dl_train{TYPENAME}' + data_name, 'rb') as f:
            dl_train = pickle.load(f)
        with open(f'{TrainConfig.dataset_dir_path}//{universe}_dl_valid{TYPENAME}' + data_name, 'rb') as f:
            dl_valid = pickle.load(f)
        with open(f'{TrainConfig.dataset_dir_path}//{universe}_dl_test{TYPENAME}' + data_name, 'rb') as f:
            dl_test = pickle.load(f)
        print("Data Loaded.")
        print(f'{TrainConfig.dataset_dir_path}//{universe}_dl_train{TYPENAME}' + data_name)

        # 核心代码

        train_loader = _init_data_loader(dl_train, shuffle=True, drop_last=True)
        valid_loader = _init_data_loader(dl_valid, shuffle=False, drop_last=False)
        test_loader = _init_data_loader(dl_test, shuffle=False, drop_last=False)

        device = TrainConfig.device
        writer = TrainConfig.writer

        # Model
        model = TrainConfig.model

        # LR

        # train_optimizer = optim.Adam(model.parameters(), lr=TrainConfig.lr, betas=(0.9, 0.999),
        #                              weight_decay=TrainConfig.weight_decay)
        train_optimizer = TrainConfig.train_optimizer
        lr_scheduler = TrainConfig.lr_scheduler

        best_valid_loss = np.Inf

        print("==" * 10 +
              f" Now is Training {TrainConfig.model_name}_{TrainConfig.seed} " + "==" * 10 + "\n")

        # 训练
        for step in range(TrainConfig.n_epoch):
            train_loss = train_epoch(train_loader, train_optimizer=train_optimizer, lr_scheduler=lr_scheduler,
                                     model=model,
                                     device=device)
            val_loss, valid_metrics = valid_epoch(valid_loader, model, device)
            test_loss, test_metrics = valid_epoch(test_loader, model, device)

            if writer is not None:
                writer.add_scalars(
                    "Valid metrics", valid_metrics, global_step=step)
                writer.add_scalars("Test metrics", test_metrics, global_step=step)
                writer.add_scalar("Train loss", train_loss, global_step=step)
                writer.add_scalar("Valid loss", val_loss, global_step=step)
                writer.add_scalar("Test loss", test_loss, global_step=step)
                writer.add_scalars("All loss Comparison",
                                   {"train loss": train_loss,
                                    "val loss": val_loss, "test loss": test_loss},
                                   global_step=step)
                writer.add_scalar(
                    "Learning rate", train_optimizer.param_groups[0]['lr'], global_step=step)

            print(
                "==" * 10 + f" {TrainConfig.model_name}_{TrainConfig.seed} Epoch {step} " + "==" * 10)
            print("Epoch %d, train_loss %.6f, valid_loss %.6f, test_loss %.6f " %
                  (step, train_loss, val_loss, test_loss))
            print("Valid Dataset Metrics performance:{}\n".format(valid_metrics))
            print("Test Dataset Metrics performance:{}\n".format(test_metrics))
            print("Learning rate :{}\n\n".format(
                train_optimizer.param_groups[0]['lr']))

            TrainConfig.logger.info(msg=f"\n===== Epoch {step} =====\ntrain loss:{train_loss}, "
                                        f"valid loss:{val_loss},test loss:{test_loss}\n"
                                        f"valid metrics:{valid_metrics}\n"
                                        f"test metrics:{test_metrics}\n"
                                        f"learning rate:{train_optimizer.param_groups[0]['lr']}\n")

            ftxt.write("===== Epoch " + str(step) + " =====\n")
            ftxt.write("train loss: " + str(train_loss) + "\n")
            ftxt.write("valid loss: {}, test loss: {}\n".format(val_loss, test_loss))
            ftxt.write("valid metrics: {}, test metrics: {}\n".format(valid_metrics, test_metrics))
            ftxt.write("learning rate: {}\n\n\n".format(train_optimizer.param_groups[0]['lr']))

            # 保存参数,新的保存策略,只保留18以上的结果
            # if step <= 10:
            #     continue

            if step % 20 == 0:
                best_valid_loss = val_loss
                model_param = copy.deepcopy(model.state_dict())
                torch.save(model_param,
                           f'{TrainConfig.model_save_path}/{TrainConfig.model_name}_epoch_{step}.pth')

        print("SAVING LAST EPOCH RESULT AS THE TEST RESULT!")
        torch.save(model.state_dict(),
                   f'{TrainConfig.model_save_path}/TEST_{TrainConfig.model_name}.pth')
        print("\n" + "==" * 10 + " Training Over " + "==" * 10)
        writer.close()

# drop="dropc7c0"
# label="o-1c-5"
# percent="975_for_test_0608"
# DATASET=f"{drop}_{label}_{percent}"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="DFT",
                        help="dataset type")
    parser.add_argument("--universe", type=str, default="csi300",
                        help="dataset type")
    parser.add_argument("--dataset", type=str, default="_2020_2023",
                        help="dataset type")
    parser.add_argument("--country", type=str, default="CN_DATA",
                        help="dataset type")
    parser.add_argument("--cuda", type=str, default="7",
                        help="dataset type")
    parser.add_argument("--seed", type=int, default=1200,
                        help="dataset type")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    set_random_seed(args.seed)
    train(args.model_name,args.dataset,args.universe,args.country,args.seed)
