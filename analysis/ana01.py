import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Function to perform summary operations
def perform_summ(d, high_pattern):
    d = d.dropna(how='all', axis=1)  # Drop columns with all NAs
    d['triplet'] = d['correct_answer_index'].shift(2).astype(str) + d['correct_answer_index'].shift(1).astype(str) + d['correct_answer_index'].astype(str)
    d = d[['orientation_index', 'correct_answer_index', 'triplet', 'key_resp.corr', 'key_resp.rt', 'thisTrialN']]
    d['type'] = np.where(d['thisTrialN'] % 2 == 1, 'Pat', 'Ran')
    d['block'] =  (d.index // 80)
    d['epoch'] =  (d.index // 400)
    d['frequency'] = np.where(d['triplet'].str.contains(high_pattern), 'High', 'Low')

    d_counts = d[d['triplet'].str.contains('NA') == False].groupby(['epoch', 'frequency', 'type', 'triplet']).size().reset_index(name='count')
    
    d_rt_epoch = d[(d['triplet'].str.contains('NA') == False) & 
                   (d['triplet'].str.contains(r"(\d)\1{2}") == False) & 
                   (d['triplet'].str.contains(r"(\d).\1") == False) &
                   (d['key_resp.corr'] == 1)].groupby(['epoch', 'frequency', 'type']).agg(RT=('key_resp.rt', 'mean')).reset_index()
    d_rt_epoch['RT'] *= 1000

    d_rt_block = d[(d['triplet'].str.contains('NA') == False) & 
                   (d['triplet'].str.contains(r"(\d)\1{2}") == False) & 
                   (d['triplet'].str.contains(r"(\d).\1") == False) &
                   (d['key_resp.corr'] == 1)].groupby(['block', 'frequency', 'type']).agg(RT=('key_resp.rt', 'mean')).reset_index()
    d_rt_block['RT'] *= 1000

    return {'epochRT': d_rt_epoch, 'frequency': d_counts, 'raw': d, 'blockRT': d_rt_block}

# Function to plot RT data
def plot_rt(typeRT, typestr, data):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=data['learn.out'][typeRT], x=typestr, y='RT', hue='frequency', style='type', markers=True, ax=ax)
    sns.scatterplot(data=data['motortest.out'][typeRT], x=typestr, y='RT', hue='frequency', style='type', markers=True, s=100, ax=ax)
    sns.scatterplot(data=data['percepttest.out'][typeRT], x=typestr, y='RT', hue='frequency', style='type', markers=True, s=100, ax=ax)
    ax.set_title(f'Reaction Time by {typestr}')
    ax.set_ylabel('Reaction Time (ms)')
    ax.set_xlabel(typestr)
    plt.legend(title='Frequency/Type')
    plt.grid(True)
    plt.show()

# Load data
droot = Path('../data/')
allfn = list(droot.glob('*.csv'))
fn = allfn[int(sys.argv[1])]
d = pd.read_csv(fn)

# Regular expression patterns
high_pattern_motor = "2.3|3.1|1.0|0.2"
high_pattern_percept = "3.2|2.1|1.0|0.3"

# Filtering and processing data
d_learn = d[d['learning_loop.thisN'].notna() & (d['learning_loop.thisTrialN'] > 4)]
d_learn['thisTrialN'] = d_learn['learning_loop.thisN']
d_motortest = d[d['motor_testing_loop.thisN'].notna() & (d['motor_testing_loop.thisTrialN'] > 4)]
d_motortest['thisTrialN'] = d_motortest['motor_testing_loop.thisN']
d_percepttest = d[d['percept_testing_loop.thisN'].notna() & (d['percept_testing_loop.thisTrialN'] > 4)]
d_percepttest['thisTrialN'] = d_percepttest['percept_testing_loop.thisTrialN']

# Perform summarization
learn_out = perform_summ(d_learn, high_pattern_motor)
motortest_out = perform_summ(d_motortest, high_pattern_motor)
percepttest_out = perform_summ(d_percepttest, high_pattern_percept)

all_outcomes = {'learn.out': learn_out, 'motortest.out': motortest_out, 'percepttest.out': percepttest_out}

# Plotting results
f_learn_epoch = plot_rt('epochRT', 'epoch', all_outcomes)
f_learn_block = plot_rt('blockRT', 'block', all_outcomes)
