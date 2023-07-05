# Standard datasets

##################################################
# INITIAL SCREENING #############################
#################################################
# Best model was norm_independant_292 so further testing will be done here
##########################################################################

# dataset_cfg1= {
#     "dems": ['26', '292', '709', '819'],
#     "name": "un_normed",
#     "num_samples": 30,
#     "grid_size_train": 120,
#     "reduced_test_size": 200,
#     "num_features": 3,
#     "normalized": None, # can be none, norm_equal, norm_all_independant, norm
#     "mask_threshold": 4000,
#     "path": "datasets/" # path will be dataset/{cfg['name']}.npy
# }

# dataset_cfg2= {
#     "dems": ['26', '292', '709', '819'],
#     "name": "all_normed_equal",
#     "num_samples": 30,
#     "grid_size_train": 120,
#     "reduced_test_size": 200,
#     "num_features": 3,
#     "normalized": "norm_equal", # can be none, norm_equal, norm_all_independant, norm
#     "mask_threshold": 4000, # choose an arbitrary size so we don't include sets of all 0's
#     "path": "datasets/" # path will be dataset/{cfg['name']}.npy
# }

# dataset_cfg3= {
#     "dems": ['26', '292', '709', '819'],
#     "name": "norm_independant",
#     "num_samples": 30,
#     "grid_size_train": 120,
#     "reduced_test_size": 200,
#     "num_features": 3,
#     "normalized": "norm_independant", # can be none, norm_equal, norm_all_independant, norm
#     "mask_threshold": 4000,
#     "path": "datasets/" # path will be dataset/{cfg['name']}.npy
# }

# config_list = [dataset_cfg1, dataset_cfg2, dataset_cfg3]


dataset_cfg1= {
    "dems": ['292'],
    "name": "norm_independant_normal",
    "num_samples": 30,
    "grid_size_train": 120,
    "reduced_test_size": 200,
    "num_features": 3,
    "normalized": "norm_independant", # can be none, norm_equal, norm_all_independant, norm
    "mask_threshold": 4000,
    "path": "datasets/", # path will be dataset/{cfg['name']}.npy
    "mask": False,
    "added_rainfall": False
}

dataset_cfg2= {
    "dems": ['292'],
    "name": "norm_independant_masked",
    "num_samples": 30,
    "grid_size_train": 120,
    "reduced_test_size": 200,
    "num_features": 4,
    "normalized": "norm_independant", # can be none, norm_equal, norm_all_independant, norm
    "mask_threshold": 4000,
    "path": "datasets/", # path will be dataset/{cfg['name']}.npy
    "mask": True,
    "added_rainfall": False
}

dataset_cfg3= {
    "dems": ['292'],
    "name": "norm_independant_rf_added",
    "num_samples": 30,
    "grid_size_train": 120,
    "reduced_test_size": 200,
    "num_features": 2,
    "normalized": "norm_independant", # can be none, norm_equal, norm_all_independant, norm
    "mask_threshold": 4000,
    "path": "datasets/", # path will be dataset/{cfg['name']}.npy
    "mask": False,
    "added_rainfall": True
}

dataset_double_check = {
    "dems": ['292'],
    "name": "double_check",
    "num_samples": 40,
    "grid_size_train": 120,
    "reduced_test_size": 200,
    "num_features": 3,
    "normalized": "norm_independant", # can be none, norm_equal, norm_all_independant, norm
    "mask_threshold": 8000,
    "path": "datasets/", # path will be dataset/{cfg['name']}.npy
    "mask": False,
    "added_rainfall": False
}
# config_list = [dataset_cfg1, dataset_cfg2, dataset_cfg3]
# config_list = [dataset_cfg3]
# config_list = [dataset_cfg3]
# config_list = [dataset_double_check]




final = dataset_cfg1= {
    "dems": ['26'],
    "name": "norm_independant_normal_final",
    "num_samples": 50,
    "grid_size_train": 128,
    "reduced_test_size": 200,
    "num_features": 3,
    "normalized": "norm_independant", # can be none, norm_equal, norm_all_independant, norm
    "mask_threshold": 9000,
    "path": "datasets/", # path will be dataset/{cfg['name']}.npy
    "mask": False,
    "added_rainfall": False
}

config_list = [final]