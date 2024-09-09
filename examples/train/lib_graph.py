import matplotlib.pyplot as plt
import matplotlib
import time

colorL = ["firebrick","sienna","olivedrab","crimson","steelblue","tomato","palegoldenrod","darkgreen","limegreen","navy","darkcyan","darkorange","brown","lightcoral","blue","red","green","yellow","purple","black"]

def style():
    plt.rcParams['axes.facecolor'] = 'white'
    matplotlib.rc('xtick', labelsize=14) 
    matplotlib.rc('ytick', labelsize=14)
    matplotlib.rcParams.update({'font.size': 14})
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'legend.fontsize': 12,'legend.handlelength': 2})
    plt.rcParams.update({'axes.labelsize': 12})
    plt.rcParams.update({'axes.titlesize': 14})

def progressBar(i,n,time_start):
    """display a progress bar"""
    prog = i/n
    delta = time.time()-time_start
    delta = delta/60
    rate = delta*prog
    #if (i%10) == 0:
    print("process %.2f time %.1f min rate %.2f\r" % (prog,delta,rate),end="\r",flush=True)
