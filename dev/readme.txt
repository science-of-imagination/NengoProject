This file explains how the /dev/ folder is organized and, more importantly how to work with it.

Put in ./utils/ any file that can be used in constructing and testing particular models
Put in ./data/ any data that might be used in constructing models and testing them; do not put model outputs
Put in ./models/ any model

To run a model you must run run.py through the system with two arguments. The first is a python file in ./models/. This file MUST provide a function, called run, which takes some parameters and returns a utils.collect.Data object. The second is a text file, called '<model-name>.params' which lists on every line the parameters to be passed to the model. It is the model's responsibility to handle any conversion issues that may come up. You can batch run a model with various parameters listed in its .params file through the run.py -l option. You can specify an alternate .params file by providing its name to run.py as a second argument.

NOTE: If you are going to modify a model in any way (besides parameters) you should save it under a different name. Such modifications means you've made a new model.

WARNING: run.py WILL NOT run unless you have created a cfg.py file. This file will let run.py know where to save model outputs. Do not push your own cfg.py file into the reporsitory.

To render the output of some model into images, simply run render_data.py, select the desired output files (yes, you can do batches). It will place the pictures in a directory whose name is identical to the data file name in the same folder as the data file.