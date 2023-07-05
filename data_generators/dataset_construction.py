import numpy as np
import h5py


def create_datasets(cfg):
    filename = 'data/better_data/val_all.h5'
    f = h5py.File(filename, "r")
    water_events_train = ['waterdepth_0', 'waterdepth_1', 'waterdepth_2', 'waterdepth_3', 'waterdepth_4', 'waterdepth_5', 'waterdepth_6', 'waterdepth_7', 'waterdepth_8', 'waterdepth_9', 'waterdepth_10', 'waterdepth_11', 'waterdepth_12', 'waterdepth_13']
    water_events_test ='waterdepth_14'

    rainfall_events_train = ['rainfall_events_0', 'rainfall_events_1', 'rainfall_events_2', 'rainfall_events_3', 'rainfall_events_4', 'rainfall_events_5', 'rainfall_events_6', 'rainfall_events_7', 'rainfall_events_8', 'rainfall_events_9', 'rainfall_events_10', 'rainfall_events_11', 'rainfall_events_12', 'rainfall_events_13']
    rainfall_events_test = 'rainfall_events_14'

    if cfg['normalized'] == None:
        for dem in cfg['dems']:

            inputs = []
            targets = []
            dset = f[dem]
            area = np.loadtxt("data/better_data/" + dem + "_dem.asc", skiprows=6)
            mask = np.array(dset['mask'])
            area *= mask
            for wd, rfe in zip(water_events_train, rainfall_events_train):
                water_depth = np.array(dset[wd])
                rainfall = np.array(dset[rfe])
                timesteps, height, width = water_depth.shape
                samples = 0
                while samples < cfg['num_samples']:
                    x, y = random_indexing(height, width, cfg['grid_size_train'])

                    if np.count_nonzero(mask[x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]) < cfg['mask_threshold']:
                        continue
        
                    for t in range(timesteps-1):
                        feature_dem = area[x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]
                        feature_rainfall = np.zeros([cfg['grid_size_train'], cfg['grid_size_train']])
                        feature_rainfall += rainfall[t]
                        feature_wd = water_depth[t, x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]
                        feature_mask = mask[x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]
                        feature_rainfall *= feature_mask
                        features = np.stack((feature_wd, feature_rainfall, feature_dem), axis=-1)
                        target = water_depth[t+1, x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]
                        inputs.append(features)
                        targets.append(target)
                    samples +=1
            inputs = np.array(inputs)
            targets = np.array(targets)
            np.save(f"datasets/inputs_{cfg['name']}_"+dem +"_.npy", inputs)
            np.save(f"datasets/targets_{cfg['name']}_"+dem +"_.npy", targets)

            # now we also need the test set which will be event14
            num_test_samples = 1
            test_samples = 0
            test_inputs = []
            test_targets = []
            test_waterdepths = np.array(dset[water_events_test])
            test_rainfall = np.array(dset[rainfall_events_test])
            timesteps, height, width = test_waterdepths.shape
            h, w = area.shape
            while test_samples < num_test_samples:
                for t in range(timesteps-1):
                    feature_dem = area[h//2 - cfg['reduced_test_size']:h//2+cfg['reduced_test_size'], w//2-cfg['reduced_test_size']:w//2+cfg['reduced_test_size']]
                    feature_rainfall = np.zeros([*feature_dem.shape])
                    feature_rainfall += test_rainfall[t]
                    feature_rainfall *= mask[h//2 - cfg['reduced_test_size']:h//2+cfg['reduced_test_size'], w//2-cfg['reduced_test_size']:w//2+cfg['reduced_test_size']]
                    feature_wd = test_waterdepths[t, h//2 - cfg['reduced_test_size']:h//2+cfg['reduced_test_size'], w//2-cfg['reduced_test_size']:w//2+cfg['reduced_test_size']]
                    # feature_mask = mask[h//2 - cfg['reduced_test_size']:h//2+cfg['reduced_test_size'], w//2-cfg['reduced_test_size']:w//2+cfg['reduced_test_size']]
                    features = np.stack((feature_wd, feature_rainfall, feature_dem), axis=-1)
                    target = test_waterdepths[t+1, h//2 - cfg['reduced_test_size']:h//2+cfg['reduced_test_size'], w//2-cfg['reduced_test_size']:w//2+cfg['reduced_test_size']]
                    test_inputs.append(features)
                    test_targets.append(target)
                test_samples +=1
            test_inputs = np.array(test_inputs)
            test_targets = np.array(test_targets)
            np.save(f"datasets/test_inputs_{cfg['name']}_"+dem +"_.npy", test_inputs)
            np.save(f"datasets/test_targets_{cfg['name']}_"+dem +"_.npy", test_targets)

    
    elif cfg['normalized'] == 'norm_equal':
        for dem in cfg['dems']:

            inputs = []
            targets = []
            dset = f[dem]
            area = np.loadtxt("data/better_data/" + dem + "_dem.asc", skiprows=6)
            mask = np.array(dset['mask'])
            area *= mask
            for wd, rfe in zip(water_events_train, rainfall_events_train):
                water_depth = np.array(dset[wd])
                rainfall = np.array(dset[rfe])
                timesteps, height, width = water_depth.shape
                samples = 0
                while samples < cfg['num_samples']:
                    x, y = random_indexing(height, width, cfg['grid_size_train'])

                    if np.count_nonzero(mask[x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]) < cfg['mask_threshold']:
                        continue
        
                    for t in range(timesteps-1):
                        feature_dem = area[x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]
                        feature_rainfall = np.zeros([cfg['grid_size_train'], cfg['grid_size_train']])
                        feature_rainfall += rainfall[t]
                        feature_wd = water_depth[t, x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]
                        feature_mask = mask[x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]
                        feature_rainfall *= feature_mask
                        features = np.stack((feature_wd, feature_rainfall, feature_dem), axis=-1)
                        target = water_depth[t+1, x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]
                        inputs.append(features)
                        targets.append(target)
                    samples +=1
            inputs = np.array(inputs)
            targets = np.array(targets)
            inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())
            targets = (targets - inputs.min()) / (inputs.max() - inputs.min())
            np.save(f"datasets/inputs_{cfg['name']}_"+dem +"_.npy", inputs)
            np.save(f"datasets/targets_{cfg['name']}_"+dem +"_.npy", targets)

            # now we also need the test set which will be event14
            num_test_samples = 1
            test_samples = 0
            test_inputs = []
            test_targets = []
            test_waterdepths = np.array(dset[water_events_test])
            test_rainfall = np.array(dset[rainfall_events_test])
            timesteps, height, width = test_waterdepths.shape
            h, w = area.shape
            while test_samples < num_test_samples:
                for t in range(timesteps-1):
                    feature_dem = area[h//2-cfg['reduced_test_size']:h//2+cfg['reduced_test_size'], w//2-cfg['reduced_test_size']:w//2+cfg['reduced_test_size']]
                    feature_rainfall = np.zeros([*feature_dem.shape])
                    feature_rainfall += test_rainfall[t]
                    feature_mask = mask[h//2 - cfg['reduced_test_size']:h//2+cfg['reduced_test_size'], w//2-cfg['reduced_test_size']:w//2+cfg['reduced_test_size']]
                    feature_rainfall *= feature_mask
                    feature_wd = test_waterdepths[t, h//2 - cfg['reduced_test_size']:h//2+cfg['reduced_test_size'], w//2-cfg['reduced_test_size']:w//2+cfg['reduced_test_size']]
                    features = np.stack((feature_wd, feature_rainfall, feature_dem), axis=-1)
                    target = test_waterdepths[t+1, h//2 - cfg['reduced_test_size']:h//2+cfg['reduced_test_size'], w//2-cfg['reduced_test_size']:w//2+cfg['reduced_test_size']]
                    test_inputs.append(features)
                    test_targets.append(target)
                test_samples +=1
            test_inputs = np.array(test_inputs)
            test_targets = np.array(test_targets)
            test_inputs = (test_inputs - inputs.min()) / (inputs.max() - inputs.min())
            test_targets = (test_targets - inputs.min()) / (inputs.max() - inputs.min())
            np.save(f"datasets/test_inputs_{cfg['name']}_"+dem +"_.npy", test_inputs)
            np.save(f"datasets/test_targets_{cfg['name']}_"+dem +"_.npy", test_targets)

    elif cfg['normalized'] == 'norm_independant' and cfg['mask'] == False and cfg['added_rainfall']==False:
        for dem in cfg['dems']:

            inputs = []
            targets = []
            dset = f[dem]
            area = np.loadtxt("data/better_data/" + dem + "_dem.asc", skiprows=6)
            mask = np.array(dset['mask'])
            area *= mask
            for wd, rfe in zip(water_events_train, rainfall_events_train):
                water_depth = np.array(dset[wd])
                rainfall = np.array(dset[rfe])
                timesteps, height, width = water_depth.shape
                samples = 0
                while samples < cfg['num_samples']:
                    x, y = random_indexing(height, width, cfg['grid_size_train'])

                    if np.count_nonzero(mask[x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]) < cfg['mask_threshold']:
                        continue
        
                    for t in range(timesteps-1):
                        feature_dem = area[x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]
                        feature_dem = (feature_dem - area.min())/(area.max() - area.min())
                        feature_rainfall = np.zeros([cfg['grid_size_train'], cfg['grid_size_train']])
                        feature_rainfall += rainfall[t]
                        feature_rainfall = (feature_rainfall/1000)/6
                        feature_wd = water_depth[t, x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]
                        feature_wd /= 8
                        feature_mask = mask[x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]
                        feature_rainfall *= feature_mask
                        features = np.stack((feature_wd, feature_rainfall, feature_dem), axis=-1)
                        target = water_depth[t+1, x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]
                        inputs.append(features)
                        targets.append(target)
                    samples +=1
            inputs = np.array(inputs)
            targets = np.array(targets)
            np.save(f"datasets/inputs_{cfg['name']}_"+dem +"_.npy", inputs)
            np.save(f"datasets/targets_{cfg['name']}_"+dem +"_.npy", targets)

            # now we also need the test set which will be event14
            num_test_samples = 1
            test_samples = 0
            test_inputs = []
            test_targets = []
            test_waterdepths = np.array(dset[water_events_test])
            test_rainfall = np.array(dset[rainfall_events_test])
            timesteps, height, width = test_waterdepths.shape
            h, w = area.shape
            while test_samples < num_test_samples:
                for t in range(timesteps-1):
                    feature_dem = area[h//2-cfg['reduced_test_size']:h//2+cfg['reduced_test_size'], w//2-cfg['reduced_test_size']:w//2+cfg['reduced_test_size']]
                    feature_dem = (feature_dem - area.min())/(area.max() - area.min())
                    feature_rainfall = np.zeros([*feature_dem.shape])
                    feature_rainfall += test_rainfall[t]
                    feature_rainfall *= mask[h//2 - cfg['reduced_test_size']:h//2+cfg['reduced_test_size'], w//2-cfg['reduced_test_size']:w//2+cfg['reduced_test_size']]
                    feature_rainfall = (feature_rainfall/1000)/6
                    feature_wd = test_waterdepths[t,h//2 - cfg['reduced_test_size']:h//2+cfg['reduced_test_size'], w//2-cfg['reduced_test_size']:w//2+cfg['reduced_test_size']]
                    feature_wd /= 8
                    # feature_mask = mask[h//2 - cfg['reduced_test_size']:h//2+cfg['reduced_test_size'], w//2-cfg['reduced_test_size']:w//2+cfg['reduced_test_size']]
                    features = np.stack((feature_wd, feature_rainfall, feature_dem), axis=-1)
                    target = test_waterdepths[t+1, h//2 - cfg['reduced_test_size']:h//2+cfg['reduced_test_size'], w//2-cfg['reduced_test_size']:w//2+cfg['reduced_test_size']]
                    test_inputs.append(features)
                    test_targets.append(target)
                test_samples +=1
            test_inputs = np.array(test_inputs)
            test_targets = np.array(test_targets)
            np.save(f"datasets/test_inputs_{cfg['name']}_"+dem +"_.npy", test_inputs)
            np.save(f"datasets/test_targets_{cfg['name']}_"+dem +"_.npy", test_targets)

    elif cfg['normalized'] == 'norm_independant' and cfg['mask'] == True and cfg['added_rainfall']==False:
        for dem in cfg['dems']:

            inputs = []
            targets = []
            dset = f[dem]
            area = np.loadtxt("data/better_data/" + dem + "_dem.asc", skiprows=6)
            mask = np.array(dset['mask'])
            area *= mask
            for wd, rfe in zip(water_events_train, rainfall_events_train):
                water_depth = np.array(dset[wd])
                rainfall = np.array(dset[rfe])
                timesteps, height, width = water_depth.shape
                samples = 0
                while samples < cfg['num_samples']:
                    x, y = random_indexing(height, width, cfg['grid_size_train'])

                    if np.count_nonzero(mask[x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]) < cfg['mask_threshold']:
                        continue
        
                    for t in range(timesteps-1):
                        feature_dem = area[x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]
                        feature_dem = (feature_dem - area.min())/(area.max() - area.min())
                        feature_rainfall = np.zeros([cfg['grid_size_train'], cfg['grid_size_train']])
                        feature_rainfall += rainfall[t]
                        feature_rainfall = (feature_rainfall/1000)/6
                        feature_wd = water_depth[t, x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]
                        feature_wd /= 8
                        feature_mask = mask[x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]
                        features = np.stack((feature_wd, feature_rainfall, feature_dem, feature_mask), axis=-1)
                        target = water_depth[t+1, x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]
                        inputs.append(features)
                        targets.append(target)
                    samples +=1
            inputs = np.array(inputs)
            targets = np.array(targets)
            np.save(f"datasets/inputs_{cfg['name']}_"+dem +"_.npy", inputs)
            np.save(f"datasets/targets_{cfg['name']}_"+dem +"_.npy", targets)

            # now we also need the test set which will be event14
            num_test_samples = 1
            test_samples = 0
            test_inputs = []
            test_targets = []
            test_waterdepths = np.array(dset[water_events_test])
            test_rainfall = np.array(dset[rainfall_events_test])
            timesteps, height, width = test_waterdepths.shape
            h, w = area.shape
            while test_samples < num_test_samples:
                for t in range(timesteps-1):
                    feature_dem = area[h//2-cfg['reduced_test_size']:h//2+cfg['reduced_test_size'], w//2-cfg['reduced_test_size']:w//2+cfg['reduced_test_size']]
                    feature_dem = (feature_dem - area.min())/(area.max() - area.min())
                    feature_rainfall = np.zeros([*feature_dem.shape])
                    feature_rainfall += test_rainfall[t]
                    feature_rainfall = (feature_rainfall/1000)/6
                    feature_wd = test_waterdepths[t,h//2 - cfg['reduced_test_size']:h//2+cfg['reduced_test_size'], w//2-cfg['reduced_test_size']:w//2+cfg['reduced_test_size']]
                    feature_wd /= 8
                    feature_mask = mask[h//2 - cfg['reduced_test_size']:h//2+cfg['reduced_test_size'], w//2-cfg['reduced_test_size']:w//2+cfg['reduced_test_size']]
                    features = np.stack((feature_wd, feature_rainfall, feature_dem, feature_mask), axis=-1)
                    target = test_waterdepths[t+1, h//2 - cfg['reduced_test_size']:h//2+cfg['reduced_test_size'], w//2-cfg['reduced_test_size']:w//2+cfg['reduced_test_size']]
                    test_inputs.append(features)
                    test_targets.append(target)
                test_samples +=1
            test_inputs = np.array(test_inputs)
            test_targets = np.array(test_targets)
            np.save(f"datasets/test_inputs_{cfg['name']}_"+dem +"_.npy", test_inputs)
            np.save(f"datasets/test_targets_{cfg['name']}_"+dem +"_.npy", test_targets)

    elif cfg['normalized'] == 'norm_independant' and cfg['mask'] == False and cfg['added_rainfall']==True:
        for dem in cfg['dems']:

            inputs = []
            targets = []
            dset = f[dem]
            area = np.loadtxt("data/better_data/" + dem + "_dem.asc", skiprows=6)
            mask = np.array(dset['mask'])
            area *= mask
            for wd, rfe in zip(water_events_train, rainfall_events_train):
                water_depth = np.array(dset[wd])
                rainfall = np.array(dset[rfe])
                timesteps, height, width = water_depth.shape
                samples = 0
                while samples < cfg['num_samples']:
                    x, y = random_indexing(height, width, cfg['grid_size_train'])

                    if np.count_nonzero(mask[x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]) < cfg['mask_threshold']:
                        continue
        
                    for t in range(timesteps-1):
                        feature_dem = area[x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]
                        feature_dem = (feature_dem - area.min())/(area.max() - area.min())
                        feature_rainfall = np.zeros([cfg['grid_size_train'], cfg['grid_size_train']])
                        feature_rainfall += rainfall[t]
                        feature_rainfall = (feature_rainfall/1000)/6
                        feature_wd = water_depth[t, x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]
                        feature_wd += feature_rainfall
                        feature_wd/=8
                        feature_mask = mask[x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]
                        feature_wd *= feature_mask
                        features = np.stack((feature_wd, feature_dem), axis=-1)
                        target = water_depth[t+1, x:x+cfg['grid_size_train'], y:y+cfg['grid_size_train']]
                        inputs.append(features)
                        targets.append(target)
                    samples +=1
            inputs = np.array(inputs)
            targets = np.array(targets)
            np.save(f"datasets/inputs_{cfg['name']}_"+dem +"_.npy", inputs)
            np.save(f"datasets/targets_{cfg['name']}_"+dem +"_.npy", targets)

            # now we also need the test set which will be event14
            num_test_samples = 1
            test_samples = 0
            test_inputs = []
            test_targets = []
            test_waterdepths = np.array(dset[water_events_test])
            test_rainfall = np.array(dset[rainfall_events_test])
            timesteps, height, width = test_waterdepths.shape
            h, w = area.shape
            while test_samples < num_test_samples:
                for t in range(timesteps-1):

                    ##### very import. When evaluating this dataset:
                    ##### remember index 1 (position 2) is rainfall which we add to the wd 
                    # and then divide the waterdepth by 8 during training. Then we multiply result by mask
                    feature_dem = area[h//2-cfg['reduced_test_size']:h//2+cfg['reduced_test_size'], w//2-cfg['reduced_test_size']:w//2+cfg['reduced_test_size']]
                    feature_dem = (feature_dem - area.min())/(area.max() - area.min())
                    feature_rainfall = np.zeros([*feature_dem.shape])
                    feature_rainfall += test_rainfall[t]
                    feature_rainfall = (feature_rainfall/1000)/6
                    feature_wd = test_waterdepths[t,h//2 - cfg['reduced_test_size']:h//2+cfg['reduced_test_size'], w//2-cfg['reduced_test_size']:w//2+cfg['reduced_test_size']]
                    feature_mask = mask[h//2 - cfg['reduced_test_size']:h//2+cfg['reduced_test_size'], w//2-cfg['reduced_test_size']:w//2+cfg['reduced_test_size']]
                    features = np.stack((feature_wd, feature_rainfall, feature_dem, feature_mask), axis=-1)
                    target = test_waterdepths[t+1, h//2 - cfg['reduced_test_size']:h//2+cfg['reduced_test_size'], w//2-cfg['reduced_test_size']:w//2+cfg['reduced_test_size']]
                    test_inputs.append(features)
                    test_targets.append(target)
                test_samples +=1
            test_inputs = np.array(test_inputs)
            test_targets = np.array(test_targets)
            np.save(f"datasets/test_inputs_{cfg['name']}_"+dem +"_.npy", test_inputs)
            np.save(f"datasets/test_targets_{cfg['name']}_"+dem +"_.npy", test_targets)
    f.close()


def random_indexing(w, h, grid_size):
    indx = np.random.randint(0, w - grid_size)
    indy = np.random.randint(0, h - grid_size)
    return indx, indy