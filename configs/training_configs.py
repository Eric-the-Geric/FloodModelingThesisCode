##################################################
# INITIAL SCREENING #############################
#################################################
# Best model was norm_independant_292 with weighted mae so further testing will be done here
##########################################################################

# training_cfg1 = {
#     'name': "train_1_mse",  'mse', 'mse', '_mse_auto', '_mae_auto', '_mse_weighted', "_mae_weighted"]
#     'path': "training_outputs/",
#     'epochs': 2,
#     'batch_size': 50,
#     'loss': 'mse',
#     'lr': 1e-3,
#     'optimizer':'adam',
#     'zero_weight': 0.8,
#     'non_zero_weight': 0.2,
#     'shuffle':False
# }


# training_cfg2 = {
#     'name': "train_1_mae",
#     'path': "training_outputs/",
#     'epochs': 2,
#     'batch_size': 50,
#     'loss': 'mae',
#     'lr': 1e-3,
#     'optimizer':'adam',
#     'zero_weight': 0.8,
#     'non_zero_weight': 0.2,
#     'shuffle': False

# }

# training_cfg3 = {
#     'name': "train_1_mse_auto",
#     'path': "training_outputs/",
#     'epochs': 2,
#     'batch_size': 50,
#     'loss': 'custom_mse_auto',
#     'lr': 1e-3,
#     'optimizer':'adam',
#     'zero_weight': 0.8,
#     'non_zero_weight': 0.2,
#     'shuffle': False

# }

# training_cfg4 = {
#     'name': "train_1_mae_auto",
#     'path': "training_outputs/",
#     'epochs': 2,
#     'batch_size': 50,
#     'loss': 'custom_mae_auto',
#     'lr': 1e-3,
#     'optimizer':'adam',
#     'zero_weight': 0.8,
#     'non_zero_weight': 0.2,
#     'shuffle': False

# }

# training_cfg5 = {
#     'name': "train_1_mse_weighted",
#     'path': "training_outputs/",
#     'epochs': 2,
#     'batch_size': 50,
#     'loss': 'custom_mse_weighted',
#     'lr': 1e-3,
#     'optimizer':'adam',
#     'zero_weight': 0.8,
#     'non_zero_weight': 0.2,
#     'shuffle': False

# }

# training_cfg6 = {
#     'name': "train_1_mae_weighted",
#     'path': "training_outputs/",
#     'epochs': 2,
#     'batch_size': 50,
#     'loss': 'custom_mae_weighted',
#     'lr': 1e-3,
#     'optimizer':'adam',
#     'zero_weight': 0.8,
#     'non_zero_weight': 0.2,
#     'shuffle': False
# }

# training_config_list = [training_cfg1, training_cfg2, training_cfg3, training_cfg4, training_cfg5, training_cfg6]

###########################################
# modifying lr *more epochs = 10 epochs; without is 2 epochs* ############################
###########################################
# training_cfg3 was found to be the best
#########################################
# training_cfg1 = {
#     'name': "train_2_custom_mae_weighted_1e-3_more_epochs",
#     'path': "training_outputs/",
#     'epochs': 10,
#     'batch_size': 50,
#     'loss': 'custom_mae_weighted',
#     'lr': 1e-3,
#     'optimizer':'adam',                                   
#     'zero_weight': 0.8,
#     'non_zero_weight': 0.2,
#     'shuffle':False
# }

# training_cfg2 = {
#     'name': "train_2_custom_mae_weighted_2e-3_more_epochs",
#     'path': "training_outputs/",
#     'epochs': 10,
#     'batch_size': 50,
#     'loss': 'custom_mae_weighted',
#     'lr': 2e-3,
#     'optimizer':'adam',
#     'zero_weight': 0.8,
#     'non_zero_weight': 0.2,
#     'shuffle':False
# }

# training_cfg3 = {
#     'name': "train_2_custom_mae_weighted_1e-4_more_epochs",
#     'path': "training_outputs/",
#     'epochs': 10,
#     'batch_size': 50,
#     'loss': 'custom_mae_weighted',
#     'lr': 1e-4,
#     'optimizer':'adam',
#     'zero_weight': 0.8,
#     'non_zero_weight': 0.2,
#     'shuffle':False
# }

# training_cfg4 = {
#     'name': "train_2_custom_mae_weighted_1e-2_more_epochs",
#     'path': "training_outputs/",
#     'epochs': 10,
#     'batch_size': 50,
#     'loss': 'custom_mae_weighted',
#     'lr': 1e-2,
#     'optimizer':'adam',
#     'zero_weight': 0.8,
#     'non_zero_weight': 0.2,
#     'shuffle':False
# }

# training_cfg5 = {
#     'name': "train_2_custom_mae_weighted_5e-4_more_epochs",
#     'path': "training_outputs/",
#     'epochs': 10,
#     'batch_size': 50,
#     'loss': 'custom_mae_weighted',
#     'lr': 5e-4,
#     'optimizer':'adam',
#     'zero_weight': 0.8,
#     'non_zero_weight': 0.2,
#     'shuffle':False
# }

# training_config_list = [training_cfg1, training_cfg2, training_cfg3, training_cfg4, training_cfg5]


# find the best weight ratio's and reduced epochs to 8
# training_cfg1 = {
#     'name': "train_3_custom_mae_weighted_zw_09_nzw_01", 
#     'path': "training_outputs/",
#     'epochs': 8,
#     'batch_size': 50,
#     'loss': 'custom_mae_weighted',
#     'lr': 1e-4,                               
#     'optimizer':'adam',
#     'zero_weight': 0.9,
#     'non_zero_weight': 0.1,
#     'shuffle':False
# }

# training_cfg2 = {
#     'name': "train_3_custom_mae_weighted_zw_08_nzw_02",
#     'path': "training_outputs/",
#     'epochs': 8,
#     'batch_size': 50,
#     'loss': 'custom_mae_weighted',
#     'lr': 1e-4,
#     'optimizer':'adam',
#     'zero_weight': 0.8,
#     'non_zero_weight': 0.2,
#     'shuffle':False
# }

# training_cfg3 = {
#     'name': "train_3_custom_mae_weighted_zw_07_nzw_03",
#     'path': "training_outputs/",
#     'epochs': 8,
#     'batch_size': 50,
#     'loss': 'custom_mae_weighted',
#     'lr': 1e-4,
#     'optimizer':'adam',
#     'zero_weight': 0.7,
#     'non_zero_weight': 0.3,
#     'shuffle':False
# }

# training_cfg4 = {
#     'name': "train_3_custom_mae_weighted_zw_06_nzw_04",
#     'path': "training_outputs/",
#     'epochs': 8,
#     'batch_size': 50,
#     'loss': 'custom_mae_weighted',
#     'lr': 1e-4,
#     'optimizer':'adam',
#     'zero_weight': 0.6,
#     'non_zero_weight': 0.4,
#     'shuffle':False
# }

# training_cfg5 = {
#     'name': "train_3_custom_mae_weighted_zw_05_nzw_05",
#     'path': "training_outputs/",
#     'epochs': 8,
#     'batch_size': 50,
#     'loss': 'custom_mae_weighted',
#     'lr': 1e-4,
#     'optimizer':'adam',
#     'zero_weight': 0.5,
#     'non_zero_weight': 0.5,
#     'shuffle':False
# }

# training_cfg6 = {
#     'name': "train_3_custom_mae_weighted_zw_04_nzw_06",
#     'path': "training_outputs/",
#     'epochs': 8,
#     'batch_size': 50,
#     'loss': 'custom_mae_weighted',
#     'lr': 1e-4,
#     'optimizer':'adam',
#     'zero_weight': 0.4,
#     'non_zero_weight': 0.6,
#     'shuffle':False
# }

# training_config_list = [training_cfg1, training_cfg2, training_cfg3, training_cfg4, training_cfg5, training_cfg6]

# for some reason 50/50 split seems to perform the best...
training_cfg1 = {
    'name': "train_4_custom_mae_weighted_1",
    'path': "training_outputs/",
    'epochs': 10,
    'batch_size': 50,
    'loss': 'custom_mae_weighted',              
    'lr': 1e-2,
    'optimizer':'adam',
    'zero_weight': 0.5,
    'non_zero_weight': 0.5,
    'shuffle':True
}
training_cfg2 = {
    'name': "train_4_custom_mae_weighted_2",
    'path': "training_outputs/",
    'epochs': 10,
    'batch_size': 50,
    'loss': 'custom_mae_weighted',
    'lr': 1e-3,
    'optimizer':'adam',
    'zero_weight': 0.5,
    'non_zero_weight': 0.5,
    'shuffle':True
}
training_cfg3 = {
    'name': "train_4_custom_mae_weighted_3",
    'path': "training_outputs/",
    'epochs': 10,
    'batch_size': 50,
    'loss': 'custom_mae_weighted',
    'lr': 1e-4,
    'optimizer':'adam',
    'zero_weight': 0.5,
    'non_zero_weight': 0.5,
    'shuffle':True
}

training_cfg4 = {
    'name': "train_4_custom_mae_weighted_4",
    'path': "training_outputs/",
    'epochs': 10,
    'batch_size': 50,
    'loss': 'custom_mae_weighted',
    'lr': 1e-2,
    'optimizer':'adam',
    'zero_weight': 0.2,
    'non_zero_weight': 0.8,
    'shuffle':True
}
training_cfg5 = {
    'name': "train_4_custom_mae_weighted_5",
    'path': "training_outputs/",
    'epochs': 10,
    'batch_size': 50,
    'loss': 'custom_mae_weighted',
    'lr': 1e-3,
    'optimizer':'adam',
    'zero_weight': 0.2,
    'non_zero_weight': 0.8,
    'shuffle':True
}
training_cfg6 = {
    'name': "train_4_custom_mae_weighted_6",
    'path': "training_outputs/",
    'epochs': 10,
    'batch_size': 50,
    'loss': 'custom_mae_weighted',
    'lr': 1e-4,
    'optimizer':'adam',
    'zero_weight': 0.2,
    'non_zero_weight': 0.8,
    'shuffle':True
}
training_cfg7 = {
    'name': "train_4_custom_mae_weighted_7_20 epochs",
    'path': "training_outputs/",
    'epochs': 20,
    'batch_size': 50,
    'loss': 'custom_mae_weighted',
    'lr': 1e-4,
    'optimizer':'adam',
    'zero_weight': 0.1,
    'non_zero_weight': 0.9,
    'shuffle':True
}

training_cfg8 = {
    'name': "train_5_custom_mae_weighted_8",
    'path': "training_outputs/",
    'epochs': 20,
    'batch_size': 50,
    'loss': 'custom_mae_weighted',
    'lr': 1e-2,
    'optimizer':'adam',
    'zero_weight': 0.2,
    'non_zero_weight': 0.8,
    'shuffle':True
}

training_cfg9 = {
    'name': "train_5_custom_mae_weighted_9",
    'path': "training_outputs/",
    'epochs': 20,
    'batch_size': 50,
    'loss': 'custom_mae_weighted',
    'lr': 1e-3,
    'optimizer':'adam',
    'zero_weight': 0.2,
    'non_zero_weight': 0.8,
    'shuffle':True
}

training_cfg10 = {
    'name': "train_5_custom_mae_weighted_10",
    'path': "training_outputs/",
    'epochs': 20,
    'batch_size': 50,
    'loss': 'custom_mae_weighted',
    'lr': 1e-3,
    'optimizer':'adam',
    'zero_weight': 0.1,
    'non_zero_weight': 0.9,
    'shuffle':True
}

training_cfg11 = {
    'name': "train_6_custom_mse_weighted_11",
    'path': "training_outputs/",
    'epochs': 20,
    'batch_size': 50,
    'loss': 'custom_mse_weighted',
    'lr': 1e-3,
    'optimizer':'adam',
    'zero_weight': 0.2,
    'non_zero_weight': 0.8,
    'shuffle':True
}

training_cfg12 = {
    'name': "train_6_custom_mae_weighted_12",
    'path': "training_outputs/",
    'epochs': 20,
    'batch_size': 50,
    'loss': 'custom_mse_weighted',
    'lr': 1e-3,
    'optimizer':'adam',
    'zero_weight': 0.3,
    'non_zero_weight': 0.9,
    'shuffle':True
}

training_cfg13 = {
    'name': "train_6_custom_mse_weighted_10",
    'path': "training_outputs/",
    'epochs': 20,
    'batch_size': 50,
    'loss': 'custom_mse_weighted',
    'lr': 1e-4,
    'optimizer':'adam',
    'zero_weight': 0.2,
    'non_zero_weight': 0.8,
    'shuffle':True
}
#training_config_list = [training_cfg1, training_cfg2, training_cfg3, training_cfg4, training_cfg5, training_cfg6]
# training_config_list = [training_cfg10, training_cfg11, training_cfg12]


# training_double_check = {
#     'name': "double_check",
#     'path': "training_outputs/",
#     'epochs': 30,
#     'batch_size': 50,
#     'loss': 'custom_mae_weighted',
#     'lr': 1e-3,
#     'optimizer':'adam',
#     'zero_weight': 0.8,
#     'non_zero_weight': 0.2,
#     'shuffle':True
# }

# training_config_list = [training_double_check]


training_f1 = {
    'name': "final_1",
    'path': "training_outputs/",
    'epochs': 20,
    'batch_size': 50,
    'loss': 'custom_mse_weighted',
    'lr': 1e-3,
    'optimizer':'adam',
    'zero_weight': 0.1,
    'non_zero_weight': 0.9,
    'shuffle':True
}

training_f2 = {
    'name': "final_2",
    'path': "training_outputs/",
    'epochs': 20,
    'batch_size': 50,
    'loss': 'custom_mse_weighted',
    'lr': 1e-3,
    'optimizer':'adam',
    'zero_weight': 0.2,
    'non_zero_weight': 0.8,
    'shuffle':True
}

training_f3 = {
    'name': "final_3",
    'path': "training_outputs/",
    'epochs': 20,
    'batch_size': 50,
    'loss': 'custom_mse_weighted',
    'lr': 1e-3,
    'optimizer':'adam',
    'zero_weight': 0.3,
    'non_zero_weight': 0.7,
    'shuffle':True
}



training_f4 = {
    'name': "final_4",
    'path': "training_outputs/",
    'epochs': 20,
    'batch_size': 50,
    'loss': 'custom_mae_weighted',
    'lr': 1e-3,
    'optimizer':'adam',
    'zero_weight': 0.1,
    'non_zero_weight': 0.9,
    'shuffle':True
}

training_f5 = {
    'name': "final_5",
    'path': "training_outputs/",
    'epochs': 20,
    'batch_size': 50,
    'loss': 'custom_mae_weighted',
    'lr': 1e-3,
    'optimizer':'adam',
    'zero_weight': 0.2,
    'non_zero_weight': 0.8,
    'shuffle':True
}

training_f6 = {
    'name': "final_6",
    'path': "training_outputs/",
    'epochs': 20,
    'batch_size': 50,
    'loss': 'custom_mae_weighted',
    'lr': 1e-3,
    'optimizer':'adam',
    'zero_weight': 0.3,
    'non_zero_weight': 0.7,
    'shuffle':True
}


training_config_list = [training_f1, training_f2, training_f3, training_f4, training_f5, training_f6]