import argparse
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
import pdb
# https://github.com/openai/baselines/blob/master/baselines/her/experiment/plot.py#L10
def smooth_reward_curve(x, y):
    halfwidth = int(np.ceil(len(x) / 60))  # Halfwidth of our smoothing convolution
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
        mode='same')
    return xsmoo, ysmoo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--files', nargs='+', help='progress.csv file')
    parser.add_argument('--labels', nargs='+')
    parser.add_argument('--title', type=str, default="FetchPush-v1")
    parser.add_argument('--yaxis', type=str, default="test/success_rate")
    parser.add_argument('--cutoff', type=int)
    parser.add_argument('--outfile', type=str, default='out.png', help='Image file')
    parser.add_argument('--smooth', action='store_true', default=False)
    pdb.set_trace()
    args = parser.parse_args()


    assert len(args.files) == len(args.labels)

    ax = None

    for file, label in zip(args.files, args.labels):
        df = pd.read_csv(file)
        if args.cutoff:
            df = df[df.epoch <= args.cutoff]
        if args.smooth:
            _, df[args.yaxis] = smooth_reward_curve(df['epoch'], df[args.yaxis])
        ax = df.plot(x='epoch', y=args.yaxis, kind='line', ax=ax, label=label)

    plt.ylabel(args.yaxis)
    plt.ylim((0., 1.))
    plt.title(args.title)

    plt.savefig(args.outfile)
    print("Wrote out to {}.".format(args.outfile))