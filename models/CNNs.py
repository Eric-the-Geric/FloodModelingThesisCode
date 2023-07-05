import tensorflow as tf


################################
### LEGACY MODELS ##############
################################


# class SimpleCNN(tf.keras.Model):
#     def __init__(self, model_config):
#         super().__init__()
#         self.config = model_config
#         self.dmodel = tf.keras.Sequential([
#             tf.keras.layers.Conv2D(self.config['num_conv_filters'], 3, padding="same", activation=tf.nn.relu),
#             tf.keras.layers.Conv2D(self.config['1x1_conv_filters'], 1, activation=tf.nn.relu),
#             tf.keras.layers.Conv2D(1, 1, activation=None)
#         ])
#         self(tf.zeros([1, 3, 3, 3]))

#     def call(self, x):
#         return self.dmodel(x)


# class SimpleCNN2(tf.keras.Model):
#     def __init__(self, model_config):
#         super().__init__()
#         self.config = model_config
#         self.dmodel = tf.keras.Sequential([
#             tf.keras.layers.Conv2D(self.config['num_conv_filters_layer1'], 3, padding="same", activation=tf.nn.relu),
#             tf.keras.layers.Conv2D(self.config['num_conv_filters_layer2'], 3, padding="same", activation=tf.nn.relu),
#             tf.keras.layers.Conv2D(self.config['num_conv_filters_layer3'], 3, padding="same", activation=tf.nn.relu),
#             tf.keras.layers.Conv2D(self.config['1x1_conv_filters_layer1'], 1, activation=tf.nn.relu),
#             tf.keras.layers.Conv2D(self.config['1x1_conv_filters_layer2'], 1, activation=tf.nn.relu),
#             tf.keras.layers.Conv2D(1, 1, activation=None, kernel_initializer=tf.keras.initializers.Zeros())
#         ])
#         self(tf.zeros([1, 3, 3, 3]))

#     def call(self, x):
#         return self.dmodel(x)
   
# class SimpleCNN3(tf.keras.Model):
#     def __init__(self, model_config):
#         super().__init__()
#         self.config = model_config
#         self.dmodel = tf.keras.Sequential([
#             tf.keras.layers.Conv2D(self.config['num_conv_filters_layer1'], 3, padding="same", activation=tf.nn.relu),
#             tf.keras.layers.Conv2D(self.config['num_conv_filters_layer2'], 3, padding="same", activation=tf.nn.relu),
#             tf.keras.layers.Conv2D(self.config['num_conv_filters_layer3'], 3, padding="same", activation=tf.nn.relu),
#             tf.keras.layers.Conv2D(self.config['1x1_conv_filters_layer1'], 1, activation=tf.nn.relu),
#             tf.keras.layers.Conv2D(self.config['1x1_conv_filters_layer2'], 1, activation=tf.nn.relu),
#             tf.keras.layers.Conv2D(1, 1, activation=tf.nn.tanh)
#         ])
#         self(tf.zeros([1, 3, 3, 3]))

#     def call(self, x):
#         return self.dmodel(x)


# Preliminary first looks showed that predicting differences was better than raw output
class BenchMark1(tf.keras.Model): # Benchmark from the evalution paper (but reduced, do to computational constraints)
    def __init__(self, model_config, dset_config):
        super().__init__()
        self.dset_config = dset_config
        self.model_config = model_config
        self.dmodel = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 5, padding="same", activation=tf.nn.relu),
            tf.keras.layers.Conv2D(32, 5, padding="same", activation=tf.nn.relu),
            tf.keras.layers.Conv2D(32, 5, padding="same", activation=tf.nn.relu),
            tf.keras.layers.Conv2D(1, 1, activation=None)
        ])
        self(tf.zeros([1, 3, 3, self.dset_config['num_features']]))

    def call(self, x):
        
        if self.model_config['diff']:
            dx = self.dmodel(x)

            if self.dset_config['mask']:
                f1, f2, f3, f4 = tf.unstack(x, axis=-1)
                f1 = tf.expand_dims(f1, -1)
                return dx + f1
            
            elif self.dset_config['added_rainfall']:
                f1, f2, = tf.unstack(x, axis=-1)
                f1 = tf.expand_dims(f1, -1)
                return dx + f1
            else:
                f1, f2, f3 = tf.unstack(x, axis=-1)
                f1 = tf.expand_dims(f1, -1)
                return dx + f1
        
        else:
            dx = self.dmodel(x)
            return dx
 
class BenchMark2(tf.keras.Model): # simple model characterized by 1x1 convolutions
    def __init__(self, model_config, dset_config):
        super().__init__()
        self.model_config = model_config
        self.dset_config = dset_config
        self.dmodel = tf.keras.Sequential([
            tf.keras.layers.Conv2D(80, 3, padding="same", activation=tf.nn.relu),
            tf.keras.layers.Conv2D(64, 1, activation=tf.nn.relu),
            tf.keras.layers.Conv2D(1, 1, activation=None)
        ])
        self(tf.zeros([1, 3, 3, self.dset_config['num_features']]))

    def call(self, x):
        
        if self.model_config['diff']:
            dx = self.dmodel(x)

            if self.dset_config['mask']:
                f1, f2, f3, f4 = tf.unstack(x, axis=-1)
                f1 = tf.expand_dims(f1, -1)
                return dx + f1
            
            elif self.dset_config['added_rainfall']:
                f1, f2, = tf.unstack(x, axis=-1)
                f1 = tf.expand_dims(f1, -1)
                return dx + f1
            else:
                f1, f2, f3 = tf.unstack(x, axis=-1)
                f1 = tf.expand_dims(f1, -1)
                return dx + f1
        
        else:
            dx = self.dmodel(x)
            return dx


# hopefully best performing model
# for the initial test 
# class DepthwiseCNN(tf.keras.Model):
#     def __init__(self, model_config, dset_config):
#         super().__init__()
#         self.model_config = model_config
#         self.dset_config = dset_config
#         self.dmodel = tf.keras.Sequential([
#             tf.keras.layers.Conv2D(self.model_config['first1x1'], 1, activation=tf.nn.relu, padding="same"),
#             tf.keras.layers.DepthwiseConv2D(3, activation=tf.nn.relu, padding="same", depth_multiplier=self.model_config['depth']),
#             tf.keras.layers.Conv2D(self.model_config['second1x1'], 1, activation=tf.nn.relu, padding="same"),
#             tf.keras.layers.Conv2D(1, 1, activation=None, padding="same", kernel_initializer=tf.zeros_initializer)
#         ])
#         self(tf.zeros([1, 3, 3, self.dset_config['num_features']]))

#     def call(self, x):
#         if self.model_config['diff']:
#             dx = self.dmodel(x)
#             f1, f2, f3 = tf.unstack(x, axis=-1)
#             f1 = tf.expand_dims(f1, -1)
#             return dx + f1
        
#         else:
#             dx = self.dmodel(x)
#             return dx
        
class DepthwiseCNN(tf.keras.Model):
    def __init__(self, model_config, dset_config):
        super().__init__()
        self.dset_config = dset_config
        self.model_config = model_config
        self.dmodel = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.model_config['first1x1'], 3, activation=tf.nn.relu, padding="same"),
            #tf.keras.layers.DepthwiseConv2D(3, activation=tf.nn.relu, padding="same", depth_multiplier=self.model_config['depth']),
            tf.keras.layers.Conv2D(self.model_config['second1x1'], 1, activation=tf.nn.relu, padding="same"),
            tf.keras.layers.Conv2D(1, 1, activation=None, padding="same", kernel_initializer=tf.zeros_initializer)
        ])
        self(tf.zeros([1, 3, 3, self.dset_config['num_features']]))

    def call(self, x):
        
        if self.model_config['diff']:
            dx = self.dmodel(x)

            if self.dset_config['mask']:
                f1, f2, f3, f4 = tf.unstack(x, axis=-1)
                f1 = tf.expand_dims(f1, -1)
                return dx + f1
            
            elif self.dset_config['added_rainfall']:
                f1, f2, = tf.unstack(x, axis=-1)
                f1 = tf.expand_dims(f1, -1)
                return dx + f1
            else:
                f1, f2, f3 = tf.unstack(x, axis=-1)
                f1 = tf.expand_dims(f1, -1)
                return dx + f1
        
        else:
            dx = self.dmodel(x)
            return dx

class IncepInspired(tf.keras.Model):
    def __init__(self, model_config, dset_config):
        super().__init__()
        self.dset_config = dset_config
        self.model_config = model_config
        self.depth = tf.keras.Sequential([
            tf.keras.layers.DepthwiseConv2D(5, activation=tf.nn.relu, padding="same", depth_multiplier=self.model_config['depth']),
            tf.keras.layers.Conv2D(self.model_config['depth_1x1'], 1, activation=tf.nn.relu, padding="same")
        ])
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.model_config['conv_filter_1'],5 , activation=tf.nn.relu, padding="same"),
            tf.keras.layers.Conv2D(self.model_config['conv_filter_2'], 5, activation=tf.nn.relu, padding="same"),
        ])

        self.conv1x1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.model_config['conv_1x1'] ,1, activation=tf.nn.relu, padding="same"),
        ])

        self.dmodel = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.model_config['first_3x3'], 5, activation=tf.nn.relu, padding="same"),
            tf.keras.layers.Conv2D(self.model_config['first1x1'], 1, activation=tf.nn.relu, padding="same"),
            tf.keras.layers.Conv2D(self.model_config['second1x1'], 1, activation=tf.nn.relu, padding="same"),
            tf.keras.layers.Conv2D(1, 1, activation=None, padding="same", kernel_initializer=tf.zeros_initializer)
        ])
        self(tf.zeros([1, 3, 3, self.dset_config['num_features']]))

    def call(self, x):
        
        if self.model_config['diff']:
            x1 = self.depth(x)
            x2 = self.conv(x)
            x3 = self.conv1x1(x)
            y = tf.concat([x1,x2,x3], axis=-1)
            
            dx = self.dmodel(y)

            if self.dset_config['mask']:
                f1, f2, f3, f4 = tf.unstack(x, axis=-1)
                f1 = tf.expand_dims(f1, -1)
                return dx + f1
            
            elif self.dset_config['added_rainfall']:
                f1, f2, = tf.unstack(x, axis=-1)
                f1 = tf.expand_dims(f1, -1)
                return dx + f1
            else:
                f1, f2, f3 = tf.unstack(x, axis=-1)
                f1 = tf.expand_dims(f1, -1)
                return dx + f1
        
        else:
            x1 = self.depth(x)
            x2 = self.conv(x)
            x3 = self.conv1x1(x)
            y = tf.stack([x1,x2,x3], axis=-1)
            dx = self.dmodel(y)
            return dx
        

##############
# Initial screening model
##################
class ComplexCNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dmodel = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, 1, activation=tf.nn.relu, padding="same"),
            tf.keras.layers.DepthwiseConv2D(3, depth_multiplier = 10, activation=tf.nn.relu, padding="same"),
            tf.keras.layers.Conv2D(8, 1, activation=tf.nn.relu, padding="same"),
            tf.keras.layers.Conv2D(1, 1, activation=None, padding="same", kernel_initializer=tf.zeros_initializer)
        ])
        self(tf.zeros([1, 3, 3, 3]))

    
    def call(self, x):
        dx = self.dmodel(x)
        f1, f2, f3 = tf.unstack(x, axis=-1)
        f1 =tf.expand_dims(f1, -1)

        return dx + f1


if __name__ == "__main__":
    

    model = ComplexCNN()
    print(model.summary())
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

    # model_config4 = {
    #     'name': 'shallow_depthwise',
    #     'depth': 1,
    #     'first1x1': 8,
    #     'second1x1': 8,
    #     'n_features': 3,
    #     'diff': False
    # }

    # model_config5 = {
    #     'name': 'medium_depthwise',
    #     'depth': 8,
    #     'first1x1': 8,
    #     'second1x1': 16,
    #     'n_features': 3,
    #     'diff': False
    # }

    # model_config6 = {
    #     'name': 'complex_depthwise',
    #     'depth': 40,
    #     'first1x1': 8,
    #     'second1x1': 64,
    #     'n_features': 3,
    #     'diff': False
    # }

    # dataset_cfg1= {
    #     "dems": ['292'],
    #     "name": "norm_independant_normal",
    #     "num_samples": 30,
    #     "grid_size_train": 120,
    #     "reduced_test_size": 200,
    #     "num_features": 3,
    #     "normalized": "norm_independant", # can be none, norm_equal, norm_all_independant, norm
    #     "mask_threshold": 4000,
    #     "path": "datasets/", # path will be dataset/{cfg['name']}.npy
    #     "mask": False,
    #     "added_rainfall": False
    # }
    
    # bm1 = BenchMark1(incep_config1, dataset_cfg1)
    # bm2 = BenchMark2(incep_config1, dataset_cfg1)
    # depth_shal =  DepthwiseCNN(model_config4, dataset_cfg1)
    # depth_med =   DepthwiseCNN(model_config5, dataset_cfg1)
    # depth_compl = DepthwiseCNN(model_config6, dataset_cfg1)
    # simple_inc = IncepInspired(incep_config1, dataset_cfg1)
    # medium_inc = IncepInspired(incep_config2, dataset_cfg1)

    
    # models =  [bm1, bm2, depth_shal, depth_med,  depth_compl, simple_inc, medium_inc]     
          
    # for model in models:
    #     print(model.summary())