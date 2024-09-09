import os, sys, gzip, random, csv, json, datetime, re
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.lib_graph as gra

# from pyneurgen.neuralnet import NeuralNet
# from pyneurgen.nodes import BiasNode, Connection
# from pyneurgen.recurrent import NARXRecurrent
def serNeural(sDay,nAhead,x0,hWeek):
    nLin = sDay.shape[0] + nAhead
    nFit = sDay.shape[0] if int(x0['obs_time']) <= 14 else int(x0['obs_time'])
    predS = getHistory(sDay,nAhead,x0,hWeek)
    weekS = [x.isocalendar()[1] for x in sDay.index]
    population = [[float(i),sDay['y'][i],float(i%7),weekS[i]] for i in range(sDay.shape[0])]
    all_inputs = []
    all_targets = []
    factorY = sDay['y'].mean()
    factorT = 1.0 / float(len(population))*factorY
    factorD = 1./7.*factorY
    factorW = 1./52.*factorY
    factorS = 4.*sDay['y'].std()
    factorH = factorY/sDay['hist'].mean()

    def population_gen(population):
        pop_sort = [item for item in population]
#        random.shuffle(pop_sort)
        for item in pop_sort:
            yield item
            
    for t,y,y1,y2 in population_gen(population):
        #all_inputs.append([t*factorT,(.5-random.random())*factorS+factorY,y1*factorD,y2*factorW])
        all_inputs.append([y1*factorD,(.5-random.random())*factorS+factorY,y2*factorW])
        all_targets.append([y])

    if False:
        plt.plot([x[0] for x in all_inputs],'-',label='targets0')
        plt.plot([x[1] for x in all_inputs],'-',label='targets1')
        plt.plot([x[2] for x in all_inputs],'-',label='targets2')
        # plt.plot([x[3] for x in all_inputs],'-',label='targets3')
        plt.plot([x[0] for x in all_targets],'-',label='actuals')
        plt.legend(loc='lower left', numpoints=1)
        plt.show()

    net = NeuralNet()
    net.init_layers(3,[10],1,NARXRecurrent(3,.6,2,.4))
    net.randomize_network()
    net.set_random_constraint(.5)
    net.set_learnrate(.1)
    net.set_all_inputs(all_inputs)
    net.set_all_targets(all_targets)
    #predS['pred'] = [item[0][0] for item in net.test_targets_activations]
    length = len(all_inputs)
    learn_end_point = int(length * .8)
    # random.sample(all_inputs,10)
    net.set_learn_range(0, learn_end_point)
    net.set_test_range(learn_end_point + 1, length - 1)
    net.layers[1].set_activation_type('tanh')

    net.learn(epochs=125,show_epoch_results=True,random_testing=False)
    mse = net.test()
    #net.save(os.environ['LAV_DIR'] + "/out/train/net.txt")

    test_positions = [item[0][0] for item in net.get_test_data()]
    all_targets1 = [item[0][0] for item in net.test_targets_activations]
    all_actuals = [item[1][0] for item in net.test_targets_activations]
    #   This is quick and dirty, but it will show the results
    plt.subplot(3, 1, 1)
    plt.plot([i for i in sDay['y']],'-')
    plt.title("Population")
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(test_positions, all_targets1, 'b-', label='targets')
    plt.plot(test_positions, all_actuals, 'r-', label='actuals')
    plt.grid(True)
    plt.legend(loc='lower left', numpoints=1)
    plt.title("Test Target Points vs Actual Points")

    plt.subplot(3, 1, 3)
    plt.plot(range(1, len(net.accum_mse) + 1, 1), net.accum_mse)
    plt.xlabel('epochs')
    plt.ylabel('mean squared error')
    plt.grid(True)
    plt.title("Mean Squared Error by Epoch")
    plt.show()

