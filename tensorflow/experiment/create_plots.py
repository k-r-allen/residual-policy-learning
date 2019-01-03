from matplotlib import pyplot as plt

import glob
import numpy as np
import os
import pandas as pd
import seaborn as sns
sns.set(style="darkgrid")



def load_df(dir_name, method, xname, yname, filename='progress.csv'):
    subdirs = [f for f in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, f))]
    dfs = []
    for subdir_name in subdirs:
        filepath = os.path.join(dir_name, subdir_name, method, filename)
        try:
            df = pd.read_csv(filepath)
            dfs.append(df)
        except:
            print("Warning: skipping", filepath)
    if len(dfs) == 0:
        return False
    return pd.concat(dfs)


def create_plot(dir_name, xname, yname, xlabel, ylabel, title, outfile, methods, labels, colors, steps_per_epoch, cutoff=50,
                residual_start_offset=0, log_plot=False, add_to_plot=None, legend_loc="lower right"):
    plt.figure()

    for method, label, color in zip(methods, labels, colors):
        df = load_df(dir_name=dir_name, method=method, xname=xname, yname=yname)
        if df is False:
            print("Warning: skipping {}.".format(method))
            continue
        df = df[df.epoch < cutoff]
        df['simulator_steps'] = df.epoch * steps_per_epoch
        if 'RPL' in label or 'Expert' in label:
            df['simulator_steps'] += residual_start_offset
        sns.lineplot(x=xname, y=yname, data=df, label=label, ci='sd', c=color)

    if log_plot:
        ax = plt.gca()
        ax.set(xscale="log")
        _, xmax = plt.xlim()
        plt.xlim((1., xmax))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if 'success_rate' in yname:
        plt.ylim((-0.1, 1.1))
        plt.yticks(np.arange(0., 1.1, 0.2))
    plt.legend(loc=legend_loc)
    if not log_plot:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    if add_to_plot:
        add_to_plot()

    plt.savefig(outfile)
    print("Wrote out to {}.".format(outfile))


def create_slippery_push_plot():
    dir_name = 'logs'
    xname = 'simulator_steps'
    yname = 'test/success_rate'
    xlabel = 'Simulator Steps'
    ylabel = 'Test Success Rate'
    title = "SlipperyPush"
    outfile = 'slippery_push_results.png'
    methods = ["FetchPushHighFriction-v0_expertexplore", "SlipperyPushInitial", "ResidualFetchPush-v0", "FetchPushHighFriction-v0"]
    labels = ['Expert Explore', 'Initial', 'RPL (Ours)', 'DDPG+HER']
    colors = ['red', 'blue', 'green', 'orange']
    steps_per_epoch = 50 * 19 * 50 * 4
    return create_plot(dir_name=dir_name, xname=xname, yname=yname, xlabel=xlabel, ylabel=ylabel, title=title, outfile=outfile,
        methods=methods, labels=labels, colors=colors, steps_per_epoch=steps_per_epoch)

def create_pickandplace_plot():
    dir_name = 'logs'
    xname = 'simulator_steps'
    yname = 'test/success_rate'
    xlabel = 'Simulator Steps'
    ylabel = 'Test Success Rate'
    title = "PickAndPlace"
    outfile = 'pickandplace_results.png'
    methods = ["FetchPickAndPlace-v1_expertexplore", "PickAndPlaceInitial", "ResidualFetchPickAndPlace-v0", "FetchPickAndPlace-v1"]
    labels = ['Expert Explore', 'Initial', 'RPL (Ours)', 'DDPG+HER']
    colors = ['red', 'blue', 'green', 'orange']
    steps_per_epoch = 50 * 19 * 50 * 4
    return create_plot(dir_name=dir_name, xname=xname, yname=yname, xlabel=xlabel, ylabel=ylabel, title=title, outfile=outfile,
        methods=methods, labels=labels, colors=colors, steps_per_epoch=steps_per_epoch)

def create_mpc_push_plot():
    dir_name = 'logs'
    xname = 'simulator_steps'
    yname = 'test/success_rate'
    xlabel = 'Simulator Steps'
    ylabel = 'Test Success Rate'
    title = "Push"
    outfile = 'mpc_push_results.png'
    methods = ["MPCPush-v0_expertexplore", "MPCPushInitial", "ResidualMPCPush-v0", "FetchPush-v1"]
    labels = ['Expert Explore', 'Initial', 'RPL (Ours)', 'DDPG+HER']
    colors = ['red', 'blue', 'green', 'orange']
    steps_per_epoch = 50 * 19 * 50 * 4
    return create_plot(dir_name=dir_name, xname=xname, yname=yname, xlabel=xlabel, ylabel=ylabel, title=title, outfile=outfile,
        methods=methods, labels=labels, colors=colors, steps_per_epoch=steps_per_epoch)

def create_noisy_hook_plot():
    dir_name = 'logs'
    xname = 'simulator_steps'
    yname = 'test/success_rate'
    xlabel = 'Simulator Steps'
    ylabel = 'Test Success Rate'
    title = "NoisyHook"
    outfile = 'noisy_hook_results.png'
    methods = ["TwoFrameHookNoisy-v0_expertexplore", "NoisyHookInitial", "TwoFrameResidualHookNoisy-v0", "TwoFrameHookNoisy-v0"]
    labels = ['Expert Explore', 'Initial', 'RPL (Ours)', 'DDPG+HER']
    colors = ['red', 'blue', 'green', 'orange']
    steps_per_epoch = 50 * 1 * 100 * 4
    cutoff = 295
    legend_loc = "upper left"
    return create_plot(dir_name=dir_name, xname=xname, yname=yname, xlabel=xlabel, ylabel=ylabel, title=title, outfile=outfile,
        methods=methods, labels=labels, colors=colors, steps_per_epoch=steps_per_epoch, cutoff=cutoff, legend_loc=legend_loc)

def create_complex_hook_plot():
    dir_name = 'logs'
    xname = 'simulator_steps'
    yname = 'test/success_rate'
    xlabel = 'Simulator Steps'
    ylabel = 'Test Success Rate'
    title = "ComplexHook"
    outfile = 'complex_hook_results.png'
    methods = ["ComplexHookTrain-v0_expertexplore", "ComplexHookInitial", "ResidualComplexHookTrain-v0", "ComplexHookTrain-v0"]
    labels = ['Expert Explore', 'Initial', 'RPL (Ours)', 'DDPG+HER']
    colors = ['red', 'blue', 'green', 'orange']
    steps_per_epoch = 50 * 1 * 100 * 4
    cutoff = 295
    legend_loc = "upper left"
    return create_plot(dir_name=dir_name, xname=xname, yname=yname, xlabel=xlabel, ylabel=ylabel, title=title, outfile=outfile,
        methods=methods, labels=labels, colors=colors, steps_per_epoch=steps_per_epoch, cutoff=cutoff, legend_loc=legend_loc)


def create_mbrl_pusher_plot():
    dir_name = 'logs'
    xname = 'simulator_steps'
    yname = 'test/returns'
    xlabel = 'Simulator Steps'
    ylabel = 'Test Returns'
    title = "MBRLPusher"
    outfile = 'mbrl_pusher_results.png'
    methods = ["OtherPusherEnv-v0_expertexplore", "ResidualOtherPusherEnv-v0", "OtherPusherEnv-v0"]
    labels = ['Expert Explore', 'RPL (Ours)', 'DDPG+HER']
    colors = ['red', 'green', 'orange']
    steps_per_epoch = 50 * 1 * 150 * 4
    cutoff = 275
    residual_start_offset = 15500

    def add_to_plot():
        xs = np.linspace(residual_start_offset, steps_per_epoch * cutoff)
        ys = np.ones_like(xs) * -74.92431339990715
        plt.plot(xs, ys, label="MBRL", color="blue", linestyle="--")
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        desired_labels_order = ['Expert Explore', 'MBRL', 'RPL (Ours)', 'DDPG+HER']
        new_handles = [handles[labels.index(label)] for label in desired_labels_order]
        ax.legend(new_handles, desired_labels_order)

    return create_plot(dir_name=dir_name, xname=xname, yname=yname, xlabel=xlabel, ylabel=ylabel, title=title, outfile=outfile,
        methods=methods, labels=labels, colors=colors, steps_per_epoch=steps_per_epoch, cutoff=cutoff, 
        residual_start_offset=residual_start_offset, log_plot=False, add_to_plot=add_to_plot)

if __name__ == "__main__":
    create_mpc_push_plot()
    create_pickandplace_plot()
    create_slippery_push_plot()
    create_noisy_hook_plot()
    create_complex_hook_plot()
    create_mbrl_pusher_plot()

