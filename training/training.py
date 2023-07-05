# imports
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from training.losses import *
from copy import deepcopy


# train models function
def train_model(model, benchmark1, benchmark2,  inputs, targets, x_test, y_test, dataset_cfg, training_cfg, model_cfg, dem_name):
    
    # Split to get training and validation data
    x_train, x_val, y_train, y_val =  train_test_split(inputs, targets, test_size=0.2, random_state=42, shuffle=training_cfg['shuffle']) # order: x_train, x_val, y_train, y_val

    #Pick  loss and optimizer
    loss_object = pick_loss(training_cfg) 
    optimizer = pick_optimizer(training_cfg)

    # compile but leave the benchmarks with default tensorflow parameters
    model.compile(optimizer=optimizer, loss=loss_object, metrics='mae')
    benchmark1.compile(optimizer='adam', loss='mse', metrics='mae')
    benchmark2.compile(optimizer='adam', loss='mse', metrics='mae')

    # Define some callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
    
    model_history = model.fit(
    x_train,
    y_train,
    batch_size=training_cfg['batch_size'],
    epochs=training_cfg['epochs'],
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    )
    benchmark1_history = benchmark1.fit(
    x_train,
    y_train,
    batch_size=1,
    epochs=2,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    )

    benchmark2_history = benchmark2.fit(
    x_train,
    y_train,
    batch_size=1,
    epochs=2,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    )

    historys = [model_history, benchmark1_history, benchmark2_history]

    plot_history(historys, training_cfg, model_cfg, dataset_cfg, dem_name)

    if dataset_cfg['added_rainfall']==False:
        benchmark_heuristic_mae, benchmark1_mae, benchmark2_mae, model_mae = evaluation_single_cell_single_timestep(model, benchmark1, benchmark2, x_test, y_test, training_cfg, dataset_cfg, model_cfg, dem_name)
        residual_model_mae = evaluation_recurrent_predictions_single_cell(model, x_test, y_test, training_cfg, dataset_cfg, model_cfg, dem_name)
        row = {"name": f"{model_cfg['name']}_{training_cfg['name']}_{dataset_cfg['name']}_{dem_name}", 
            "heuristic_mae": benchmark_heuristic_mae, 
            "benchmark1_mae": benchmark1_mae, 
            "benchmark2_mae": benchmark2_mae, 
            "model_mae": model_mae,
            "residual_model_mae": residual_model_mae
            }
        
        return row
    elif dataset_cfg['added_rainfall']==True:
        benchmark_heuristic_mae, benchmark1_mae, benchmark2_mae, model_mae = evaluation_single_cell_single_timestep_rf_added(model, benchmark1, benchmark2, x_test, y_test, training_cfg, dataset_cfg, model_cfg, dem_name)
        residual_model_mae = evaluation_recurrent_predictions_single_cell_rf_added(model, x_test, y_test, training_cfg, dataset_cfg, model_cfg, dem_name)
        row = {"name": f"{model_cfg['name']}_{training_cfg['name']}_{dataset_cfg['name']}_{dem_name}", 
            "heuristic_mae": benchmark_heuristic_mae, 
            "benchmark1_mae": benchmark1_mae, 
            "benchmark2_mae": benchmark2_mae, 
            "model_mae": model_mae,
            "residual_model_mae": residual_model_mae
            }
        
        return row


def benchmark_shift(data):
    shifted_data = np.roll(data, shift=1, axis=0)
    shifted_data[0,...] = 0
    return shifted_data[..., 0]


def pick_loss(config):
    if config['loss'] == 'mse':
        return 'mse'
    
    elif config['loss'] == 'mae':
        return['mae']
    
    elif config['loss'] == 'custom_mse_auto':
        return CustomMSE_Count_Zero()
    
    elif config['loss'] == 'custom_mae_auto':
        return CustomMAE_Count_Zero()
    
    elif config['loss'] == 'custom_mse_weighted':
        return CustomMSE_Own_Weight(zero_weight=config['zero_weight'], non_zero_weight=config['non_zero_weight'])
    
    elif config['loss'] == 'custom_mae_weighted':
        return CustomMAE_Own_Weight(zero_weight=config['zero_weight'], non_zero_weight=config['non_zero_weight'])
    
    else:
        raise ValueError("We didn't creat a loss object..., maybe check config")
        


def pick_optimizer(config):
    if config['optimizer'] == 'adam':
        return tf.keras.optimizers.Adam(config['lr'])


def plot_history(historys, training_cfg, model_cfg, dataset_cfg, dem_name):
    # summarize history for loss
    # order is:  [model_history, benchmark1_history, benchmark2_history]
    for history in historys:
        plt.plot(np.log10(history.history['loss']))
        plt.plot(np.log10(history.history['val_loss']))
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['m_train', 'm_val', 'b1_train', 'b1_val', 'b2_train', 'b2_val'], loc='upper left')
    plt.savefig(f"{training_cfg['path']}{model_cfg['name']}_{training_cfg['name']}_{dataset_cfg['name']}_{dem_name}_training")
    plt.close()



def evaluation_single_cell_single_timestep(model, benchmark1, benchmark2, x_test, y_test, training_cfg, dataset_cfg, model_cfg, dem_name):
    y_buffer = deepcopy(y_test)
    x_buffer = deepcopy(x_test)
    # for eval ...
    y_flat = y_buffer.flatten()
    model_predictions = model.predict(x_buffer)
    benchmark1_predictions = benchmark1.predict(x_buffer)
    benchmark2_predictions = benchmark2.predict(x_buffer)
    benchmark_heuristic_predictions = x_test[..., 0]#benchmark_shift(x_buffer) # needs to be last to not f data WHY IS IT SHIFTING TWICE???
    
    # doubles as evaluation
    benchmark_heuristic_mae = mean_absolute_error(y_flat, benchmark_heuristic_predictions.flatten())
    benchmark1_mae = mean_absolute_error(y_flat, benchmark1_predictions.flatten())
    benchmark2_mae = mean_absolute_error(y_flat, benchmark2_predictions.flatten())
    model_mae = mean_absolute_error(y_flat, model_predictions.flatten())
    
    
    t, h, w, c = x_test.shape
    plt.figure(figsize=(10, 10))
    plt.suptitle("Evalutation for single time step")
    for _ in range(1, 5):

        while True:
            # make sure we don't get cringe cells with no water
            rand_cell_x = np.random.randint(0, h)
            rand_cell_y = np.random.randint(0, w)
            if np.sum(y_test[:, rand_cell_x, rand_cell_y]) > 0:
                break
        plt.subplot(2, 2, _)
        plt.plot(range(t),model_predictions[:, rand_cell_x, rand_cell_y, 0], 'r-', label="model_pred")
        plt.plot(range(t),benchmark1_predictions[:, rand_cell_x, rand_cell_y, 0], 'g-', label="bm1_preed")
        plt.plot(range(t),benchmark2_predictions[:, rand_cell_x, rand_cell_y, 0], 'b-', label="bm2_pred")
        plt.plot(range(t),benchmark_heuristic_predictions[:, rand_cell_x, rand_cell_y], 'y-', label="bmH_pred")
        plt.plot(range(t),y_test[:, rand_cell_x, rand_cell_y], 'o-', label="Target")
        plt.xticks(range(t))
        plt.xlabel("timesteps")
        plt.ylabel("Values")
        plt.legend()
        plt.title(f"Prediction and Targets for {t} timestep x={rand_cell_x} y={rand_cell_y}")
    plt.savefig(f"{training_cfg['path']}{model_cfg['name']}_{training_cfg['name']}_{dataset_cfg['name']}_{dem_name}_random_cell_single_timestep")
    plt.close()

    return benchmark_heuristic_mae, benchmark1_mae, benchmark2_mae, model_mae

def evaluation_single_cell_single_timestep_rf_added(model, benchmark1, benchmark2, x_test, y_test, training_cfg, dataset_cfg, model_cfg, dem_name):
    y_buffer = deepcopy(y_test)
    x_buffer = deepcopy(x_test)
    x_heuristic = deepcopy(x_test)
    x_heuristic/=8
    x_wd, x_rf, x_dem, x_mask = tf.unstack(x_buffer, axis=-1)
    x_wd += x_rf
    x_wd *= x_mask
    x_wd /=8
    x_buffer = np.stack([x_wd, x_dem], axis=-1)
    # for eval ...
    y_flat = y_buffer.flatten()
    model_predictions = model.predict(x_buffer)
    benchmark1_predictions = benchmark1.predict(x_buffer)
    benchmark2_predictions = benchmark2.predict(x_buffer)
    benchmark_heuristic_predictions = benchmark_shift(x_heuristic) # needs to be last to not f data
    
    # doubles as evaluation
    benchmark_heuristic_mae = mean_absolute_error(y_flat, benchmark_heuristic_predictions.flatten())
    benchmark1_mae = mean_absolute_error(y_flat, benchmark1_predictions.flatten())
    benchmark2_mae = mean_absolute_error(y_flat, benchmark2_predictions.flatten())
    model_mae = mean_absolute_error(y_flat, model_predictions.flatten())
    
    
    t, h, w, c = x_test.shape
    while True:
        # make sure we don't get cringe cells with no water
        rand_cell_x = np.random.randint(0, h)
        rand_cell_y = np.random.randint(0, w)
        if np.sum(y_test[:, rand_cell_x, rand_cell_y]) > 0:
            break
    plt.figure(figsize=(10, 5))
    plt.plot(range(t), model_predictions[:, rand_cell_x, rand_cell_y, 0], 'r-', label="model_pred")
    plt.plot(range(t), benchmark1_predictions[:, rand_cell_x, rand_cell_y, 0], 'g-', label="bm1_preed")
    plt.plot(range(t), benchmark2_predictions[:, rand_cell_x, rand_cell_y, 0], 'b-', label="bm2_pred")
    plt.plot(range(t), benchmark_heuristic_predictions[:, rand_cell_x, rand_cell_y], 'y-', label="bmH_pred")
    plt.plot(range(t), y_test[:, rand_cell_x, rand_cell_y], 'o-', label="Target")
    plt.xticks(range(t))
    plt.xlabel("timesteps")
    plt.ylabel("Values")
    plt.legend()
    plt.title(f"Prediction and Targets for {t} timestep x={rand_cell_x} y={rand_cell_y}")
    plt.savefig(f"{training_cfg['path']}{model_cfg['name']}_{training_cfg['name']}_{dataset_cfg['name']}_{dem_name}_random_cell_single_timestep")
    plt.close()

    return benchmark_heuristic_mae, benchmark1_mae, benchmark2_mae, model_mae

def evaluation_recurrent_predictions_single_cell(model, x_test, y_test, training_cfg, dataset_cfg, model_cfg, dem_name):
    residual_inputs = deepcopy(x_test)
    x_heuristic = deepcopy(x_test)
    #x_heuristic/=8
    t, h, w, c = x_test.shape
    # print(residual_inputs.shape)
    residual_predictions = [residual_inputs[0, ...,0]]
    data_in = residual_inputs[0, ...]
    for i in range(t-1):
        data_in = data_in[None,...]
        data_out = model(data_in)
        next_data = residual_inputs[i+1,...]
        if c == 4:
            wd, rf, dem, mask = tf.unstack(next_data, axis=-1)
            data_in = np.stack([data_out[0,...,0], rf, dem, mask], axis=-1)
        else:
            wd, rf, dem = tf.unstack(next_data, axis=-1)
            data_in = np.stack([data_out[0,...,0], rf, dem], axis=-1)
        residual_predictions.append(data_out[0,...,0])
    
    #benchmark_heuristic_predictions = x_test[..., 0]#benchmark_shift(x_heuristic)
    residual_predictions = np.array(residual_predictions)
    y_buffer = deepcopy(y_test)
    y_flat = y_buffer.flatten()
    residual_model_mae = mean_absolute_error(y_flat, residual_predictions.flatten())

    plt.figure(figsize=(10, 10))
    plt.suptitle("Recurrent preditions")
    for _ in range(1, 5):
        plt.subplot(2, 2, _)
        while True:
            rand_cell_x = np.random.randint(0, h)
            rand_cell_y = np.random.randint(0, w)
            if np.sum(y_test[:, rand_cell_x, rand_cell_y]) > 0:
                break
        plt.plot(range(t), residual_predictions[:, rand_cell_x, rand_cell_y], 'r-', label="Predictions")
        plt.plot(range(t), y_test[:, rand_cell_x, rand_cell_y], 'b-', label="Targets")
        plt.xticks(range(t))
        plt.xlabel("timestep")
        plt.ylabel("Values")
        plt.legend()
        plt.title(f"Prediction and Targets for {t} timesteps x={rand_cell_x} y={rand_cell_y}")
    plt.savefig(f"{training_cfg['path']}{model_cfg['name']}_{training_cfg['name']}_{dataset_cfg['name']}_{dem_name}_random_cell_residual")
    plt.close()
    return residual_model_mae



def evaluation_recurrent_predictions_single_cell_rf_added(model, x_test, y_test, training_cfg, dataset_cfg, model_cfg, dem_name):
    residual_inputs = deepcopy(x_test)
    t, h, w, c = x_test.shape
    # print(residual_inputs.shape)

    residual_predictions = [residual_inputs[0, ...,0]]
    data_in = residual_inputs[0, ...]
    wd, rf, dem, mask = tf.unstack(data_in, axis=-1)
    wd+=rf
    wd*=mask
    wd/=8
    data_in = np.stack([wd, dem], axis=-1)
    for i in range(t-1):
        data_in = data_in[None,...]
        data_out = model(data_in)
        new_wd = data_out[0,...,0]
        next_data = residual_inputs[i+1,...]
        filler, rf, dem, mask = tf.unstack(next_data, axis=-1)
        new_wd+=tf.cast(rf, dtype=tf.float32)
        new_wd*=tf.cast(mask, dtype=tf.float32)
        new_wd/=8
        data_in = np.stack([new_wd, dem], axis=-1)
        residual_predictions.append(new_wd)
    
    benchmark_heuristic_predictions = benchmark_shift(residual_inputs)
    residual_predictions = np.array(residual_predictions)
    y_buffer = deepcopy(y_test)
    y_flat = y_buffer.flatten()
    residual_model_mae = mean_absolute_error(y_flat, residual_predictions.flatten())

    while True:
        rand_cell_x = np.random.randint(0, h)
        rand_cell_y = np.random.randint(0, w)
        if np.sum(y_test[:, rand_cell_x, rand_cell_y]) > 0:
            break
    plt.figure(figsize=(10, 5))
    plt.plot(range(t), residual_predictions[:, rand_cell_x, rand_cell_y], 'r-', label="Predictions")
    plt.plot(range(t), y_test[:, rand_cell_x, rand_cell_y], 'b-', label="Targets")
    plt.xticks(range(t))
    plt.xlabel("timestep")
    plt.ylabel("Values")
    plt.legend()
    plt.title(f"Prediction and Targets for {t} timesteps x={rand_cell_x} y={rand_cell_y}")
    plt.savefig(f"{training_cfg['path']}{model_cfg['name']}_{training_cfg['name']}_{dataset_cfg['name']}_{dem_name}_random_cell_residual")
    plt.close()
    return residual_model_mae