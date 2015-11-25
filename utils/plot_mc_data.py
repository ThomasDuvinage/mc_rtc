#!/usr/bin/env python

import csv
import itertools
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import sys

REF_JOINT_ORDER = [
    "RLEG_JOINT0", "RLEG_JOINT1", "RLEG_JOINT2", "RLEG_JOINT3", "RLEG_JOINT4", "RLEG_JOINT5",
    "LLEG_JOINT0", "LLEG_JOINT1", "LLEG_JOINT2", "LLEG_JOINT3", "LLEG_JOINT4", "LLEG_JOINT5",
    "CHEST_JOINT0", "CHEST_JOINT1", "HEAD_JOINT0", "HEAD_JOINT1",
    "RARM_JOINT0", "RARM_JOINT1", "RARM_JOINT2", "RARM_JOINT3", "RARM_JOINT4", "RARM_JOINT5", "RARM_JOINT6", "RARM_JOINT7",
    "LARM_JOINT0", "LARM_JOINT1", "LARM_JOINT2", "LARM_JOINT3", "LARM_JOINT4", "LARM_JOINT5", "LARM_JOINT6", "LARM_JOINT7",
    "RHAND_JOINT0", "RHAND_JOINT1", "RHAND_JOINT2", "RHAND_JOINT3", "RHAND_JOINT4",
    "LHAND_JOINT0", "LHAND_JOINT1", "LHAND_JOINT2", "LHAND_JOINT3", "LHAND_JOINT4"]

def usage():
    print "{0} [.log]".format(sys.argv[0])

if len(sys.argv) < 2:
    usage()
    sys.exit(0)

data = {}
with open(sys.argv[1]) as fd:
    reader = csv.DictReader(fd, delimiter=';')
    for row in reader:
        for k in reader.fieldnames:
            if k in data:
                data[k].append(float(row[k]))
            else:
                data[k] = [float(row[k])]
    for k in data:
        data[k] = np.array(data[k])
    print "Read data from log file"

if not len(data):
    print "Failed to read data from provided file"
    sys.exit(0)

assert('t' in data)
data['t'] = data['t'] - data['t'][0]
plt.xlim(xmin = 0, xmax = np.max(data['t']))

def plot_stance_index_fig(ax, scale, cc):
    if scale:
        ymin, ymax = ax.get_ylim()
        ymin += 0.1*(ymax-ymin)
        ymax -= 0.1*(ymax-ymin)
        nmax = np.max(data['stance_index'])
        ax.plot(data['t'], data['stance_index']*(ymax-ymin)/nmax + ymin, color = cc.next(), alpha = 0.5)
    else:
        ax.plot(data['t'], data['stance_index'], color = cc.next(), alpha = 0.5)
        ax.set_ylabel('stance')

def plot_torque_fig(joint_names, ax, cc):
    if isinstance(joint_names, str):
        joint_names = [joint_names]
    ymin, ymax = ax.get_ylim()
    if ymax == 1.0:
        ymax = -1e6
    if ymin == 0.0:
        ymin = 1e6
    for j in joint_names:
        if j in REF_JOINT_ORDER:
            tauc_idx = 'taucIn' + str(REF_JOINT_ORDER.index(j))
            if tauc_idx in data:
                ax.plot(data['t'], data[tauc_idx], label='{0} torque'.format(j), color = cc.next())
                ymin = min(ymin, np.min(data[tauc_idx]))
                ymax = max(ymax, np.max(data[tauc_idx]))
    if ymin < 0:
        ymin = 1.2*ymin
    else:
        ymin = 0.8*ymin
    if ymax > 0:
        ymax = 1.2*ymax
    else:
        ymax = 0.8*ymax
    ax.set_ylim(ymin = ymin, ymax = ymax)
    ax.set_ylabel('Torque')

def plot_encoder_fig(joint_names, ax, cc):
    if isinstance(joint_names, str):
        joint_names = [joint_names]
    ymin, ymax = ax.get_ylim()
    if ymax == 1.0:
        ymax = -1e6
    if ymin == 0.0:
        ymin = 1e6
    for j in joint_names:
        if j in REF_JOINT_ORDER:
            qIn_idx = 'qIn' + str(REF_JOINT_ORDER.index(j))
            if qIn_idx in data:
                ax.plot(data['t'], data[qIn_idx], label='{0} encoder'.format(j), color = cc.next())
                ymin = min(ymin, np.min(data[qIn_idx]))
                ymax = max(ymax, np.max(data[qIn_idx]))
    if ymin < 0:
        ymin = 1.2*ymin
    else:
        ymin = 0.8*ymin
    if ymax > 0:
        ymax = 1.2*ymax
    else:
        ymax = 0.8*ymax
    ax.set_ylim(ymin = ymin, ymax = ymax)
    ax.set_ylabel('Encoder')

def plot_command_fig(joint_names, ax, cc):
    if isinstance(joint_names, str):
        joint_names = [joint_names]
    ymin, ymax = ax.get_ylim()
    if ymax == 1.0:
        ymax = -1e6
    if ymin == 0.0:
        ymin = 1e6
    for j in joint_names:
        if j in REF_JOINT_ORDER:
            qOut_idx = 'qOut' + str(REF_JOINT_ORDER.index(j))
            if qOut_idx in data:
                ax.plot(data['t'], data[qOut_idx], label='{0} command'.format(j), color = cc.next())
                ymin = min(ymin, np.min(data[qOut_idx]))
                ymax = max(ymax, np.max(data[qOut_idx]))
    if ymin < 0:
        ymin = 1.2*ymin
    else:
        ymin = 0.8*ymin
    if ymax > 0:
        ymax = 1.2*ymax
    else:
        ymax = 0.8*ymax
    ax.set_ylim(ymin = ymin, ymax = ymax)
    ax.set_ylabel('Command')

def plot_error_fig(joint_names, ax, cc):
    if isinstance(joint_names, str):
        joint_names = [joint_names]
    ymin, ymax = ax.get_ylim()
    if ymax == 1.0:
        ymax = -1e6
    if ymin == 0.0:
        ymin = 1e6
    for j in joint_names:
        if j in REF_JOINT_ORDER:
            qIn_idx = 'qIn' + str(REF_JOINT_ORDER.index(j))
            qOut_idx = 'qOut' + str(REF_JOINT_ORDER.index(j))
            if qIn_idx in data and qOut_idx in data:
                y_data = data[qOut_idx] - data[qIn_idx]
                ax.plot(data['t'], y_data, label='{0} error'.format(j), color = cc.next())
                ymin = min(ymin, np.min(y_data))
                ymax = max(ymax, np.max(y_data))
    if ymin < 0:
        ymin = 1.2*ymin
    else:
        ymin = 0.8*ymin
    if ymax > 0:
        ymax = 1.2*ymax
    else:
        ymax = 0.8*y_max
    ax.set_ylim(ymin = ymin, ymax = ymax)
    ax.set_ylabel('Error')

def prep_ax():
    fig, ax = plt.subplots()
    color_cycler = itertools.cycle(['r','g','b','y','k','c','m','orange'])
    ax2 = ax.twinx()
    return ax,ax2,color_cycler

def plot_torque(joint_names):
    ax, ax2, cc = prep_ax()
    plot_torque_fig(joint_names, ax, cc)
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
    plot_stance_index_fig(ax2, False, cc)
    plt.show()

def plot_encoder(joint_names):
    ax, ax2, cc = prep_ax()
    plot_encoder_fig(joint_names, ax, cc)
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
    plot_stance_index_fig(ax2, False, cc)
    plt.show()

def plot_command(joint_names):
    ax, ax2, cc = prep_ax()
    plot_command_fig(joint_names, ax, cc)
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
    plot_stance_index_fig(ax2, False, cc)
    plt.show()

def plot_error(joint_names):
    ax, ax2, cc = prep_ax()
    plot_error_fig(joint_names, ax, cc)
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
    plot_stance_index_fig(ax2, False, cc)
    plt.show()

def plot_torque_error(joint_names):
    ax, ax2, cc = prep_ax()
    plot_error_fig(joint_names, ax, cc)
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
    plot_torque_fig(joint_names, ax2, cc)
    plot_stance_index_fig(ax2, True, cc)
    ax2.legend(bbox_to_anchor=(0., -.1, 1., -1.02), loc=3, ncol=4, mode="expand", borderaxespad=0.)
    plt.show()

def welcome():
    print "Available functions:"
    print "- plot_torque(joint_names)"
    print "- plot_encoder(joint_names)"
    print "- plot_command(joint_names)"
    print "- plot_error(joint_names)"
    print "- plot_torque_error(joint_names)"

welcome()
