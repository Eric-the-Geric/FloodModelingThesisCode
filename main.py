# import dataset_generator which calls config giles
            # --> outputs datasets in dataset dir
# import model configs to create models
# train models -> output loss plots, benchmark comparisons + model weights
import pandas as pd
from configs.dataset_configs import config_list as dset_configs
from data_generators.dataset_construction import create_datasets
import numpy as np
from models.CNNs import BenchMark1, BenchMark2, DepthwiseCNN, IncepInspired
from configs.model_configs import  model_config_list # incep_config_list as
from training.training import train_model
from configs.training_configs import training_config_list
import matplotlib.pyplot as plt
def main(mk_dataset=False):
    try:
        df = pd.read_csv("training_outputs/evaluations.csv")
        print("success")
    except:
        print("new_df... is this what I expected??")
        columns = ["name", "heuristic_mae", "benchmark1_mae", "benchmark2_mae", "model_mae", "residual_model_mae"]
        df = pd.DataFrame(columns=columns)
        df.to_csv("training_outputs/evaluations.csv", sep=',',decimal='.', header=True, index=False)
    
    if mk_dataset:
        for dset_cfg in dset_configs:
            print("making dataset, stay tuned")
            create_datasets(dset_cfg)
            print("done making dataset. check the datasets directory")
    else:
        print("no dataset created")

    print("datasets loaded, continueing grid search for best dataset/training/model combo")
    for dset_cfg in dset_configs:
        for dem_name in dset_cfg['dems']:
            for training_cfg in training_config_list:
                for model_cfg in model_config_list:
                    print(model_cfg['name'], training_cfg['name'], dset_cfg['name'])
                    inputs = np.load(f'{dset_cfg["path"]}inputs_{dset_cfg["name"]}_{dem_name}_.npy')
                    targets = np.load(f'{dset_cfg["path"]}targets_{dset_cfg["name"]}_{dem_name}_.npy')
                    x_test = np.load(f'{dset_cfg["path"]}test_inputs_{dset_cfg["name"]}_{dem_name}_.npy')

                    y_test = np.load(f'{dset_cfg["path"]}test_targets_{dset_cfg["name"]}_{dem_name}_.npy')
                    # model = DepthwiseCNN(model_cfg, dset_cfg)
                    model = DepthwiseCNN(model_cfg, dset_cfg)
                    benchmark1 = BenchMark1(model_cfg, dset_cfg)
                    benchmark2 = BenchMark2(model_cfg, dset_cfg)
                    row = train_model(model, benchmark1, benchmark2,  inputs, targets, x_test, y_test, dset_cfg, training_cfg, model_cfg, dem_name)
                    df = pd.DataFrame([row])
                    df.to_csv("training_outputs/evaluations.csv", mode='a', sep=',',decimal='.', header=False, index=False)

def plot_shift(x_test, y_test):
    t, h, w, c = x_test.shape
    print(x_test.shape)
    benchmark_heuristic_predictions = benchmark_shift(x_test)
    while True:
        # make sure we don't get cringe cells with no water
        rand_cell_x = np.random.randint(0, h)
        rand_cell_y = np.random.randint(0, w)
        if np.sum(y_test[:, rand_cell_x, rand_cell_y]) > 0:
            break
    plt.figure(figsize=(10, 5))
    plt.plot(range(t),x_test[:, rand_cell_x, rand_cell_y, 0], 'r-', label="input")
    plt.plot(range(t),benchmark_heuristic_predictions[:, rand_cell_x, rand_cell_y], 'y-', label="bmH_pred")
    plt.plot(range(t),y_test[:, rand_cell_x, rand_cell_y], 'o-', label="Target")
    plt.xticks(range(t))
    plt.xlabel("timesteps")
    plt.ylabel("Values")
    plt.legend()
    plt.title(f"Prediction and Targets for {t} timestep x={rand_cell_x} y={rand_cell_y}")

    plt.show()
def benchmark_shift(data):
    shifted_data = np.roll(data, shift=1, axis=0)
    shifted_data[0,...] = 0
    return shifted_data[..., 0]

if __name__ == "__main__":
    main(mk_dataset=False)
