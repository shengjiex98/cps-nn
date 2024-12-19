'''
    this file is for the input parameters of reinforcement leraning of binary NAS
    1. Architecture of NN;
'''

layers = 7 # '7' is an example for number of layers
controller_parameter = {
    # the number of layer, the number of neurons for each layer
    'layer':[[1,2,3]],
    'convolution_type':[[1,2,3]],
    'conv_type_second':[[1,2,3]],
    'num_channel':[[16,32,64,128,256],[16,32,64,128,256],[16,32,64,128,256],[16,32,64,128,256]],
    'num_children_per_episode': 1,
    'hidden_units': 35,
    'max_episodes': 10,
    'power_constraint': 50000
}

