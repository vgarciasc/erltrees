import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pylab import rcParams
from gym_dssat_pdi.envs.utils import utils as dssat_utils
import matplotlib.ticker as plticker
figsize = (8, 6)
rcParams['figure.figsize'] = figsize
import seaborn as sns

sns.set_context('paper', font_scale=2)
import pandas as pd
import pickle
from pprint import pprint
import itertools
import pdb

font_scale = 10


def load_data(path):
    with open(path, 'rb') as f_:
        histories = pickle.load(f_)
    return histories


def get_actions(history_dict, mode):
    action_name_dic = {'fertilization': 'anfer', 'irrigation': 'amir'}
    assert mode in action_name_dic
    action_name = action_name_dic[mode]
    action_dic = {}
    for key in [*history_dict]:
        action_dic[key] = []
        for episode_history in history_dict[key]['action']:
            action_dic[key].append([action[action_name] for action in episode_history])
    return action_dic


def get_rewards(history_dict):
    reward_dic = {}
    for key in [*history_dict]:
        reward_dic[key] = []
        for episode_history in history_dict[key]['reward']:
            reward_dic[key].append(episode_history)
    return reward_dic


def plot_actions(history_dict, mode, saving_path, keys=None):
    if keys is None:
        keys = [*history_dict]
    episode_lengths = []
    all_actions = get_actions(history_dict, mode)
    replications = 0
    for index, key in enumerate(keys):
        for actions_ in all_actions[key]:
            if index == 0:
                replications += 1
            episode_lengths.append(len(actions_))
    max_episode_length = max(episode_lengths)
    steps = range(1, max_episode_length + 1)
    dict_for_df = {'step': [], 'action': [], 'policy': []}
    for index, key in enumerate(keys):
        actions = all_actions[key]
        all_actions_list = []
        all_steps = []
        for actions_ in actions:
            nan_to_fill = [np.nan for _ in range(max_episode_length - len(actions_))]
            actions_.extend(nan_to_fill)
            all_actions_list.extend(actions_)
            all_steps.extend(steps)
        dict_for_df['step'].extend(all_steps)
        dict_for_df['policy'].extend(np.repeat(key, len(all_steps)))
        dict_for_df['action'].extend(all_actions_list)
    df = pd.DataFrame.from_dict(dict_for_df)
    df = df[df['action'] > 0]
    # colors = sns.color_palette('cool_r', len([*history_dict]))[:-1]
    colors = [(0.4980392156862745, 0.5019607843137255, 1.0),
              (0.24705882352941178, 0.7529411764705882, 1.0),
              (0.7490196078431373, 0.25098039215686274, 1.0),
              ][1:]
    fig, ax = plt.subplots(figsize=(8, 6))
    if mode == 'fertilization':
        ylocator = 10
    else:
        ylocator = 5
    sns.histplot(df, x='step', y='action', hue='policy', binwidth=(10, ylocator), palette=colors, ax=ax)
    ax.set_xlabel('day of simulation')
    # ax.set_xlim(1, max_episode_length)
    if mode == 'fertilization':
        ylabel = 'fertilizer quantity (kg/ha)'
        title = f'Nitrogen fertilizer applications ({replications} episodes)'
    else:
        ylabel = 'irrigated water (L/m2)'
        title = f'Irrigations ({replications} episodes)'
    ax.set_ylabel(ylabel)
    move_legend(ax, "upper left")
    ax.legend_.set_title(None)
    fig.set_size_inches(8, 6)
    plt.tight_layout()
    ax.yaxis.set_label_coords(-.07, .5)
    ax.xaxis.set_label_coords(.5, -.08)
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=20.0))
    ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=10.0))
    if mode == 'irrigation':
        ax.yaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
        ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=5))
    else:
        ax.yaxis.set_major_locator(plticker.MultipleLocator(base=20.0))
        ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=10.0))
    plt.subplots_adjust(top=0.95)
    plt.title(title)
    plt.savefig(saving_path)


def plot_rewards(history_dict, mode, dos_cut=155, y_logscale=False, y_label=None, x_logscale=False, x_label=None, q_high=.95,
                 q_low=.05, saving_path=None, title=None, quantile_range_legend=True):
    reward_dict = get_rewards(history_dict)
    policy_names = [*reward_dict]
    if saving_path is None:
        saving_path = f'fertilization_policy_rewards.pdf'
    # colors = sns.color_palette('cool_r', len([*reward_dict]))
    colors = [(0.4980392156862745, 0.5019607843137255, 1.0),
              (0.24705882352941178, 0.7529411764705882, 1.0),
              (0.7490196078431373, 0.25098039215686274, 1.0),
              ]
    dashes = ['dashed', 'solid', 'dotted']
    dashes = itertools.cycle(dashes)
    markers = ['*', 'X', '^']
    markers = itertools.cycle(markers)
    legend_elements = []
    fig, ax = plt.subplots(figsize=(8, 6))
    replications = 0
    for index, (policy_name, color) in enumerate(zip(policy_names, colors)):
        dash = next(dashes)
        rewards = reward_dict[policy_name]
        rewards_ = []
        for reward in rewards:
            if index == 0:
                replications += 1
            if len(reward) > dos_cut:
                rewards_.append(reward[:dos_cut])
        rewards = np.asarray(rewards_).cumsum(axis=1)
        _, horizon = rewards.shape
        marker = next(markers)
        alpha_q = .1
        linewidth = 2
        if index == 0:
            zorder = 2
        else:
            zorder = 1
        quantiles_cum_reward_low = np.quantile(rewards, q=q_low, axis=0)
        quantiles_cum_reward_high = np.quantile(rewards, q=q_high, axis=0)
        mean_regret = np.mean(rewards, axis=0)
        x = np.array(range(1, horizon + 1))
        line = ax.plot(x, mean_regret, color=color, linewidth=linewidth, linestyle=dash,
                       zorder=zorder + 2)
        x_points_step = horizon // 10
        x_points = range(x_points_step, horizon, x_points_step)
        common_x_indexes = [i for i, x in enumerate(x) if x in x_points]
        y_points = mean_regret[common_x_indexes]
        points = ax.plot(x_points, y_points, color=color, marker=marker, zorder=zorder + 2,
                         linestyle='', markersize=10, lw=3)
        legend_element = Line2D([None], [None], lw=2, label=policy_name, linestyle=dash, marker=marker,
                                linewidth=linewidth, color=color, markersize=10)
        legend_elements.append(legend_element)
        ax.fill_between(x=x, y1=quantiles_cum_reward_low, y2=quantiles_cum_reward_high, color=color,
                        zorder=zorder - 1, alpha=alpha_q)
        ax.plot(x, quantiles_cum_reward_low, color=color, linewidth=1, linestyle=dash, zorder=zorder)
        ax.plot(x, quantiles_cum_reward_high, color=color, linewidth=1, linestyle=dash, zorder=zorder)
    # loc = plticker.MultipleLocator(base=1)
    # ax.xaxis.set_major_locator(loc)
    if x_logscale:
        ax.set_xscale('log')
    if y_logscale:
        ax.set_yscale('log')
    if x_label is None:
        x_label = 'day of simulation'
        if x_logscale:
            x_label = f'{x_label[:-1]} log(t)'
    ax.set_xlabel(x_label)
    if y_label is None:
        y_label = 'cumulated return'
    if y_logscale:
        y_label = f'log {y_label}'
    ax.set_ylabel(y_label)
    if quantile_range_legend:
        patch = Patch(facecolor='black', edgecolor=None, label=f'[{q_low:.02f},{q_high:.02f}] quantile range',
                      alpha=.2)
        legend_elements.append(patch)
    # ax.grid(axis='both')
    ax.legend(handles=legend_elements, loc='best')
    if title is None:
        title = f'Policy returns ({replications} episodes)'
    plt.title(title)
    ax.tick_params(axis='both', which='major', pad=3)
    ax.yaxis.set_label_coords(-.07, .5)
    ax.xaxis.set_label_coords(.5, -.08)
    ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=10.0))
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=20.0))
    if mode == 'irrigation':
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.tight_layout()
    plt.savefig(saving_path)
    plt.close(fig)


def get_statistics(history_dict, mode):
    if mode == 'fertilization':
        features_subset = ['topwt', 'grnwt', 'pcngrn', 'cumsumfert', 'cleach']
        action_name = 'anfer'
    else:
        features_subset = ['topwt', 'grnwt', 'totir', 'runoff', 'cleach']
        action_name = 'amir'
    features = [*features_subset, 'efficiency', 'duration', 'napp']

    state_features_dic = {key: {feature: [] for feature in features} for key in [*history_dict]}

    for key in [*history_dict]:
        for repetition_index, repetition in enumerate(history_dict[key]['state']):
            last_state = repetition[-1]
            for feature in features_subset:
                values = last_state[feature]
                state_features_dic[key][feature].append(values)

    for key in [*history_dict]:
        grnwt_rep = state_features_dic[key]['grnwt']
        grnwt_0_rep = state_features_dic['null']['grnwt']
        if mode == 'fertilization':
            rate_rep = state_features_dic[key]['cumsumfert']
        else:
            grnwt_rep, grnwt_0_rep = 10 * grnwt_rep, 10 * grnwt_0_rep
            rate_rep = state_features_dic[key]['totir']
        for grnwt, grnwt_0, cumsumfert in zip(grnwt_rep, grnwt_0_rep, rate_rep):
            if cumsumfert == 0:
                value = np.nan
            else:
                value = (grnwt - grnwt_0) / cumsumfert
            state_features_dic[key]['efficiency'].append(value)

    for key in [*history_dict]:
        for applications in history_dict[key]['action']:
            applications = [application[action_name] for application in applications]
            state_features_dic[key]['napp'].append((np.asarray(applications) > 0).sum())
            state_features_dic[key]['duration'].append(len(applications))

    df = make_df_from_dict(state_features_dic)
    return df

def get_growing_stage_occurences(history_dict):
    growing_stage_dict = {key: {} for key in [*history_dict]}
    for key in [*history_dict]:
        istage_dic = {}
        for repetition_index, repetition in enumerate(history_dict[key]['state']):
            istages = [state['istage'] for state in repetition]
            istage_atoms = np.unique(istages)
            for istage_atom in istage_atoms:
                first_occurence_index = istages.index(istage_atom)
                first_occurence_das = first_occurence_index + 1
                if istage_atom not in istage_dic:
                    istage_dic[istage_atom] = []
                istage_dic[istage_atom].append(first_occurence_das)
        for istage_atom in [*istage_dic]:
            growing_stage_dict[key][f'{istage_atom}_mean'] = np.mean(istage_dic[istage_atom])
            growing_stage_dict[key][f'{istage_atom}_std'] = np.std(istage_dic[istage_atom], ddof=1)
    df = pd.DataFrame.from_dict(growing_stage_dict, orient='index')
    istage_order = [7, 8, 9, 1, 2, 3, 4, 5, 6]
    column_order = []
    for istage in istage_order:
        column_order.append(f'{istage}_mean')
        column_order.append(f'{istage}_std')
    df = df[column_order]
    return df

def check_dap(history_dict):
    for key in [*history_dict]:
        sowing_das = []
        for repetition_index, repetition in enumerate(history_dict[key]['state']):
            daps = [state['dap'] for state in repetition]
            sowing_das.append(daps.index(1))
        print(f'min sowing das : {max(sowing_das)} for {key}')
        print(f'max sowing das : {min(sowing_das)} for {key}')


def make_df_from_dict(state_features_dic):
    df_dic = {}
    keys = [*state_features_dic]
    for key in keys:
        features = [*state_features_dic[key]]
        features_key = [f'{feature}_{key}' for feature in features]
        for feature, feature_key in zip(features, features_key):
            values = state_features_dic[key][feature]
            if feature == 'pcngrn':
                values = 100 * np.asarray(values)
                feature_key += '_pct'
            df_dic[feature_key] = values
    df = pd.DataFrame.from_dict(df_dic, orient='columns')
    args = np.argsort(df.columns)
    df = df.iloc[:, args]
    return df


def move_legend(ax, new_loc, **kws):
    """
    from https://github.com/mwaskom/seaborn/issues/2280
    """
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)


if __name__ == '__main__':
    mode = 'fertilization'
    # mode = 'irrigation'
    print(f'###########################\n## MODE: {mode} ##\n###########################')
    for dir in [f'./figures/{mode}']:
        dssat_utils.make_folder(dir)
    history_dict = load_data(path=f'./output/{mode}/evaluation_histories.pkl')
    # check_dap(history_dict)
    df_stages = get_growing_stage_occurences(history_dict)
    df_stages.transpose().round(0).to_csv(f'./output/{mode}/growing_stages.csv', index_label='istage')
    action_keys = ['ppo', 'expert']
    plot_actions(history_dict=history_dict, mode=mode, saving_path=f'./figures/{mode}/{mode}Applications.pdf',
                 keys=action_keys)
    plot_rewards(history_dict=history_dict, mode=mode, quantile_range_legend=False,
                 saving_path=f'./figures/{mode}/{mode}Rewards.pdf')
    df_stats = get_statistics(history_dict=history_dict, mode=mode)
    df_stats.describe().round(1).to_csv(f'./output/{mode}/advanced_evaluation.csv')