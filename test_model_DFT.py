import os
import torch
os.environ['CUDA_LAUNCH_BLOCKING'] = "7"
os.environ['CUDA_VISIBLE_DEVICES'] = "7"  # 程序可见的GPU
cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

import pickle
import argparse
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.data import Sampler

from DFT import DFT
from my_lr_scheduler import ChainedScheduler

# 创建一个模型名称到类的映射字典
model_classes = {
    "DFT": DFT,
}
class TESTCONFIG:
    def __init__(self,model_name="DFT", GPU=0, universe='csi800', seed=11032,
                 dataset_country="CN_DATA",dataset="_2020_2023_step_8_c1c3"):
        self.model_name = model_name
        self.GPU = GPU
        self.universe = universe
        self.seed=seed
        self.dataset_country=dataset_country
        self.dataset=dataset

        self.model_param_path = f"./model_params/{universe}/TEST_{model_name}{seed}_{universe}{dataset}.pth"
        # 确保model_param_path被提供或使用默认值s
        # 加载checkpoint标志
        self.load_check = "Checkpoint" in os.path.basename(self.model_param_path).split("_")
        self.dataset_dir_path = f"./DATASETS/{dataset_country}/{universe}/2020_2023"

        self.metrics_path = f"./metrics/{self.universe}/{self.model_name}_{self.seed}"
        self.labels_pred_path = f"./label_pred/{self.universe}/{self.model_name}_{self.seed}"

        if not os.path.exists(self.model_param_path):
            print(self.model_param_path)
            raise FileNotFoundError("Model parameters file does not exist!")

        if not os.path.exists(self.metrics_path):
            os.makedirs(self.metrics_path)

        if not os.path.exists(self.labels_pred_path):
            os.makedirs(self.labels_pred_path)

        # 模型设置
        self.seq_len = 8
        self.d_feat = 158
        self.d_model = 256
        self.n_head = 4
        self.dropout = 0.5
        self.gate_input_start_index = 158
        self.gate_input_end_index = 221
        self.beta = 10
        self.device = torch.device(f"cuda:{self.GPU}" if torch.cuda.is_available() else "cpu")

        self.model=model_classes[model_name](d_model=self.d_model, d_feat=self.d_feat, seq_len=self.seq_len,
                           t_nhead=self.n_head, S_dropout_rate=self.dropout, beta=self.beta).to(self.device)
        # 加载模型参数
        if self.load_check:
            checkpoint = torch.load(self.model_param_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_param"])
        else:
            self.model.load_state_dict(torch.load(self.model_param_path, map_location=self.device))


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


def _init_data_loader(data, shuffle=True, drop_last=False):
    sampler = DailyBatchSamplerRandom(data, shuffle)
    data_loader = DataLoader(data, sampler=sampler, drop_last=drop_last)
    return data_loader


def test(model_name="DFT", GPU=0, universe='csi800', seed=11032,
                 dataset_country="CN_DATA",dataset="_2020_2023_step_5_c1c3"):
    TestConfig = TESTCONFIG(model_name, GPU, universe, seed,dataset_country,dataset)

    file_path = f"./DATASETS/CN_DATA/{universe}/{universe}_dl_test{dataset}.pkl"
    with open(file_path, 'rb') as f:
        dl_test = pickle.load(f)
    print("Data Loaded.")

    test_loader = _init_data_loader(dl_test, shuffle=False, drop_last=False)

    device = TestConfig.device

    # Model
    model = TestConfig.model
    seed = TestConfig.seed
    model_name = TestConfig.model_name

    preds = []
    ic = []
    ric = []
    labels = []

    print("==" * 10 + f"Now is Testing {model_name}_{seed}" + "==" * 10 + "\n")

    model.eval()
    for data in test_loader:
        data = torch.squeeze(data, dim=0)
        feature = data[:, :, 0:-1].to(device)
        label = data[:, -1, -1]
        with torch.no_grad():
            pred = model(feature.float()).detach().cpu().numpy()
        preds.append(pred.ravel())
        labels.append(label.ravel())

        daily_ic, daily_ric = calc_ic(pred, label.detach().numpy())
        ic.append(daily_ic)
        ric.append(daily_ric)

    predictions = pd.Series(np.concatenate(
        preds), name="score", index=dl_test.get_index())
    labels = pd.Series(np.concatenate(labels), name="label",
                       index=dl_test.get_index())

    metrics = {
        'IC': np.mean(ic),
        'ICIR': np.mean(ic) / np.std(ic),
        'RIC': np.mean(ric),
        'RICIR': np.mean(ric) / np.std(ric)
    }
    print("\nTest Dataset Metrics performance:{}\n".format(metrics))

    # 保存结果
    with open(os.path.join(TestConfig.metrics_path, f"{model_name}_{seed}_test_result.txt"), "w") as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value}\n")

    return predictions, labels, metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="DFT",
                        help="dataset type")
    parser.add_argument("--GPU", type=int, default=0,
                        help="dataset type")
    parser.add_argument("--universe", type=str, default="csi300",
                        help="dataset type")
    parser.add_argument("--seed", type=int, default=1200,
                        help="dataset type")
    parser.add_argument("--dataset_country", type=str, default="CN_DATA",
                        help="dataset type")
    parser.add_argument("--dataset", type=str, default="_2020_2023",
                        help="dataset type")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    predictions, labels, _ = test(args.model_name, args.GPU, args.universe, args.seed,args.dataset_country,args.dataset)
    if not os.path.exists("./label_pred"):
        os.mkdir("./label_pred")
    with open(f"./label_pred/{args.universe}/{args.model_name}{args.seed}_pred_{args.dataset}.pkl", "wb") as f:
        pickle.dump(predictions, f)
    # print(predictions)
    with open(f"./label_pred/{args.universe}/{args.model_name}{args.seed}_labels_{args.dataset}.pkl", "wb") as f:
        pickle.dump(labels, f)
