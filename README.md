# FloodModelingThesisCode
All the code used to generate  and evaluate data for my thesis



The config directory contains all the configurations used to generate the results. You can make a configuration file by copying the structure you see. Important that you add that to the associated  list at the bottom of the file which main will loop through to generate results.

Exploring_valh5 is a jupyter notebook where preliminary and small scale testing was done. It was also used to generate some of the plots you see in the thesis.

data_analysis is a jypyter notebook where the majority of results were generated (plots, tables etc).

main.py is the main file... It packages the code and performs grid search using the configuration file lists that it imports. It takes 1 parameter: makedataset: bool. If you haven't generated a dataset, this allows you to generate a dataset based on a configuration and immediately perform a grid search on it after it constructs it. It is constructed based on the content the file in 'data_generators'. Important to note, you need to have the data I had access to in order to use pretty much any of the code. But it is too big to put in this github repo.

The models are imported from the cnn.py file located in the 'models' directory. They have all been set up to use configuration files for some ability to adjust the parameter count.

once you succesfully run main, the outputs will all go into 'training_outputs'. I recommend you add this directory to gitignore once you clone it.

Training the models occurs from the 'train.py' file located in the 'training' directory. This contains all the code for training the models and generating plots for performance. This is also where the heuristic is located (called benchmark shift).

Good luck sifting through this!!!!!
