
# Note, cannot do many depths if many 1x1 convolutions

# ***** Found medium_depthwise_diff the best
# model_config1 = {
#     'name': 'shallow_depthwise_diff',
#     'depth': 1,
#     'first1x1': 8,
#     'second1x1': 8,
#     'n_features': 3,
#     'diff': True
# }

# model_config2 = {
#     'name': 'medium_depthwise_diff',
#     'depth': 8,
#     'first1x1': 8,
#     'second1x1': 16,
#     'n_features': 3,
#     'diff': True
# }

# model_config3 = {
#     'name': 'complex_depthwise_diff',
#     'depth': 40,
#     'first1x1': 8,
#     'second1x1': 32,
#     'n_features': 3,
#     'diff': True
# }

# model_config4 = {
#     'name': 'shallow_depthwise_no_diff',
#     'depth': 1,
#     'first1x1': 8,
#     'second1x1': 8,
#     'n_features': 3,
#     'diff': False
# }

# model_config5 = {
#     'name': 'medium_depthwise_no_diff',
#     'depth': 8,
#     'first1x1': 8,
#     'second1x1': 16,
#     'n_features': 3,
#     'diff': False
# }

# model_config6 = {
#     'name': 'complex_depthwise_no_diff',
#     'depth': 40,
#     'first1x1': 8,
#     'second1x1': 64,
#     'n_features': 3,
#     'diff': False
# }

# model_config_list = [model_config1, model_config2, model_config3, model_config4, model_config5, model_config6]


model_config1 = {
    'name': 'medium_depthwise_diff',
    'depth': 8,
    'first1x1': 8,
    'second1x1': 16,
    'n_features': 3,
    'diff': True,
    'mask': False
}

# model_config_list = [model_config1]



# incep_config1 = {
#     'name': 'simple_incep',
#     'depth': 1,
#     'depth_1x1': 8,
#     'conv_filter_1': 5,
#     'conv_filter_2': 8,
#     'conv_1x1': 8,
#     'first1x1': 8,
#     'first_3x3': 8,
#     'second1x1': 16,
#     'diff': True,
    
# }

# incep_config2 = {
#     'name': 'medium_incep',
#     'depth': 8,
#     'depth_1x1': 12,
#     'conv_filter_1': 32,
#     'conv_filter_2': 12,
#     'conv_1x1': 12,
#     'first_3x3': 32,
#     'first1x1': 8,
#     'second1x1': 16,
#     'diff': True,
    
# }

# incep_config_list = [incep_config1, incep_config2]


model_double_check = {
    'name': 'double_check',
    'depth': 1,
    'first1x1': 40,
    'second1x1': 68,
    'n_features': 3,
    'diff': True
}

model_config_list = [model_double_check]