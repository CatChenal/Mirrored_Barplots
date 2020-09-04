import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from collections import Counter


def set_spines(ax, cat_len, side='left'):   
    if side =='left':
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_zorder(10)
    else:
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_zorder(10)

    ax.spines['top'].set_position(('data',cat_len+.25))
    ax.spines['top'].set_color('w')

    
def set_ax_lims(ax, max_lim, cat_len, side='left'):
    lim = (max_lim, 0) if side == 'left' else (0, max_lim)
    plt.xlim(lim)
    plt.ylim((0, cat_len))


def set_ax_ticks(ax, tx_pos, tx_lbls, side, tx_kwargs):
    ax.xaxis.set_ticks_position('top')
    plt.xticks(tx_pos, tx_lbls, **tx_kwargs)
    
    if side == 'left':
        ax.get_xticklabels()[-1].set_weight('bold')
        ax.get_xticklines()[-1].set_markeredgewidth(0)
        ax.yaxis.set_ticks_position('right')
    else:
        ax.get_xticklabels()[0].set_weight('bold')
        ax.get_xticklines()[1].set_markeredgewidth(0)
        ax.yaxis.set_ticks_position('left')
    
    plt.yticks([])

    
def plot_ax_series_rects(ax, paired_series, fc):
    """paired_series = (smaller, larger) or (larger,)
    """
    H,h = 0.8, 0.55
    alfa = 0.4
    
    if len(paired_series) == 2:
        s1, s2 = paired_series
        only1 = False
    else:
        s1 = paired_series[0]
        only1 = True
        h, H = H, h
        alfa /= 5

    for i in range(len(s1)):
        if not only1:
            value = s2[i]
            p = patches.Rectangle(
                (0, i + (1-H)/2.0), value, H,
                fill=True, lw=0,
                transform=ax.transData,
                facecolor=fc,
                alpha=alfa/4)
            ax.add_patch(p)

        value = s1[i]
        p = patches.Rectangle(
            (0, i + (1-h)/2.0), value, h,
            fill=True, lw=0,
            transform=ax.transData,
            facecolor=fc,
            alpha=alfa)
        ax.add_patch(p)
    
    ax.grid(color='w')


def annotate_top_pairs(ax, side,
                       cat_len,
                       paired_series,
                       notes=('smaller','larger')):
    only1 = False
    arrowprops = dict(arrowstyle="-",
                      connectionstyle="angle,angleA=0,angleB=90,rad=0")

    if len(paired_series) == 1:
        only1 = True
        s0 = paired_series[0]
    else:
        s0 = paired_series[0]
        s1 = paired_series[1]
    
    #ax.spines['top'].set_position(('data',cat_len+.25))
    yloc = 0.96*cat_len
    
    if side == 'left':
        ax.annotate(notes[0],
                    xy=(0.85*s0[-1], yloc),
                    xycoords='data',
                    xytext=(-50, -25),
                    textcoords='offset points',
                    ha='right',
                    fontsize=10,
                    fontweight='bold',
                    arrowprops=arrowprops)
        
        if not only1:
            ax.annotate(notes[1], xy=(0.9*s1[-1], yloc),
                        xycoords='data',
                        xytext=(-40, -3),
                        textcoords='offset points',
                        ha='right',
                        fontsize= 10,
                        fontweight='bold',
                        arrowprops=arrowprops)
    else:
        ax.annotate(notes[0], xy=(0.9*s0[-1], yloc),
                    xycoords='data',
                    xytext=(+50, -25),
                    textcoords='offset points',
                    ha='left',
                    fontsize=10,
                    fontweight='bold',
                    arrowprops=arrowprops)
        if not only1:
            ax.annotate(notes[1], xy=(0.9*s1[-1], yloc),
                        xycoords='data',
                        xytext=(+40, -3),
                        textcoords='offset points',
                        ha='left',
                        fontsize=10,
                        fontweight='bold',
                        arrowprops=arrowprops)


def add_caption(ax,
                caption_id,
                cat_len,
                caption_hdr,
                caption_txt,
                caption_args):
    
    cnt = Counter(caption_hdr)
    hdr_lines = cnt['\n'] + 1
    
    hdr_fontsize = caption_args.get('hdr_fontsize', 16)
    txt_fontsize = caption_args.get('txt_fontsize', 11)
    x_pos_offset = caption_args.get('x_pos_offset', 0)
    y_pos_pct = caption_args.get('y_pos_pct', 0.7)
    
    if caption_id == 0:
        xt = ax.get_xticks()[0]
        xpos = xt - x_pos_offset
        ma = 'left'
        
    else:
        xt = ax.get_xticks()[1]
        xpos = xt + x_pos_offset
        ma = 'right'

    ypos_hdr = cat_len * y_pos_pct
    ax.text(xpos, ypos_hdr,
            caption_hdr,
            fontsize=hdr_fontsize,
            multialignment=ma,
            va='top')
    
    ypos_txt = cat_len * (y_pos_pct - hdr_lines * 0.05)
    ax.text(xpos, ypos_txt,
            caption_txt,
            fontsize=txt_fontsize,
            multialignment=ma,
            va='top')


def set_ylabels(axes_pair,
                category,
                fontsize=10,
                ofs=0.5):
    """
    To obtain y-axis label in between the two y spines.
    """
    import textwrap

    long_cats = [(i, textwrap.wrap(s, 25,
                                 break_long_words=False))
                for i, s in enumerate(category) if len(s)>25]

    for cat in long_cats:
        category[cat[0]] = '\n'.join(c for c in cat[1])
    
    d_len = []
    for i, d in enumerate(category):
        x1,y1 = axes_pair[0].transData.transform_point((0, i+ofs))
        x2,y2 = axes_pair[1].transData.transform_point((0, i+ofs))
        
        f = plt.gcf()
        x,y = f.transFigure.inverted().transform_point(((x1+x2)/2, y1))
        
        plt.text(x, y, d,
                 transform=f.transFigure,
                 fontsize=fontsize,
                 ha='center', va='center')
        
        d_len.append(len(d))

    ofs_ws=0.65
    ws = max(10/(max(d_len) + 3*ofs_ws), 0.36)
    f.subplots_adjust(wspace=ws)
    
    
def plot_paired_categories(categories,
                           left_series, right_series,
                           xt_pos, xt_lbls,
                           face_cols=['red','blue'],
                           equal_lims=True,
                           y_lbl_ofs=0.5,
                           pair_labels=['larger set', 'smaller set'],
                           annotate_side='both',
                           caption_ax=None,
                           caption=['Caption Header', 'caption text'],
                           caption_kwargs=None,
                           tx_kwargs=None,
                           fig_style=None,
                           save_fig=False,
                           save_as=None):
    """
    Attempt to generalize the implementation by
    Nicolas P. Rougier of New York Times graphics, 2007
    (http://www.nytimes.com/imagepages/2007/07/29/health/29cancer.graph.web.html).
    Plot single or double paired categories
    Inputs:
    -------
    :param: categories (list)
    :param: left_ , right_series: tuple of at most 2 lists: (smaller, larger) or (larger,)
    :param: xt_pos (2-tuple of list), xt_lbls (2-tuple of list): position, labels
    :param: face_cols: bar color for each of the series
    :param: equal_lims (bool)
    :param: pair_labels (list) default=['larger set', 'smaller set'],
    :param: caption_ax (axis index) default=None,
    :param: caption (list) default=['Caption Header', 'caption text'],
    :param: caption_kwargs (dict) x_ticks fontstyle
    :param: save_fig (bool)
    :param: tx_kwargs (dict) x_ticks fontstyle
    :param: fig_style (dict)
    :param: save_fig=False, save_as=None
    """
    # Checks
    if not isinstance(categories, list):
        raise ValueError("`categories` must be a list.")
        
    cat_len = len(categories)
    #  non-zero cat:
    if not cat_len:
        raise ValueError("`categories` is an empty list.")
        
    #   number of series on each side, min=1, max=2:
    #   to do
    only1 = False
    
    if equal_lims:
        mx_left = max(max(left_series[0]), max(left_series[1]))
        mx_right = max(max(right_series[0]), max(right_series[1]))
        mx_lims = max(mx_left, mx_right)
        sep = xt_pos[1][1] - xt_pos[1][0]

        if mx_lims not in xt_pos[0]:
            if mx_lims >= xt_pos[1][-1] + sep:
                xt_pos[0].insert(0, mx_lims)
                xt_lbls[0].insert(0, str(mx_lims))
        elif mx_lims not in xt_pos[1]:
            if mx_lims <= xt_pos[0][0] - sep:
                xt_pos[1].append(mx_lims)
                xt_lbls[1].append(str(mx_lims))
    
    # --- plot construction -------------------------------
    fig = plt.figure(**fig_style)
    
    if tx_kwargs is None:
        tx_kwargs = {'fontsize': 11}
        
    # ---left data ---
    axes_left  = plt.subplot(121)
    # Keep only top and right spines
    s = 'left'
    set_spines(axes_left, cat_len, side=s)
    set_ax_lims(axes_left, mx_lims, cat_len, side=s)
    set_ax_ticks(axes_left,
                 xt_pos[0], xt_lbls[0],
                 s, tx_kwargs)   
    plot_ax_series_rects(axes_left,
                         left_series,
                         fc=face_cols[0])
    if annotate_side == 'both' or annotate_side == s:
        annotate_top_pairs(axes_left, s,
                           cat_len,
                           left_series,
                           notes=pair_labels)  

    # --- right data ----------------------------------------
    axes_right = plt.subplot(122, sharey=axes_left)
    s = 'right'
    set_spines(axes_right, cat_len, side=s)
    set_ax_lims(axes_left, mx_lims, cat_len, side=s)
    set_ax_ticks(axes_right,
                 xt_pos[1], xt_lbls[1],
                 s, tx_kwargs)

    plot_ax_series_rects(axes_right,
                         right_series,
                         fc=face_cols[1])
    
    if annotate_side == 'both' or annotate_side == s:
        annotate_top_pairs(axes_right, s,
                           cat_len,
                           right_series,
                           notes=pair_labels)

    # --- Y axis & caption ---------------------------------
    set_ylabels((axes_left, axes_right),
                categories,
                fontsize=11,
                ofs=y_lbl_ofs)

    if caption_ax is not None:
        if caption_ax == 0:
            cax = axes_left
        else:
            cax = axes_right
            
        add_caption(cax,
                    caption_ax,
                    cat_len,
                    caption[0],
                    caption[1],
                    caption_kwargs)
    
    if save_fig:
        if save_as is not None:
            plt.savefig(save_as)
        else:
            print('Figure not saved: save_as=None.')

# cases:: new cases
rougier_data = dict(diseases=["Kidney Cancer", "Bladder Cancer", "Esophageal Cancer",
                              "Ovarian Cancer", "Liver Cancer", "Non-Hodgkin's\nlymphoma",
                              "Leukemia", "Prostate Cancer", "Pancreatic Cancer",
                              "Breast Cancer", "Colorectal Cancer", "Lung Cancer"],
                    women_deaths = [6_000, 5_500, 5_000, 20_000, 9_000, 12_000,
                                    13_000, 0, 19_000, 40_000, 30_000, 70_000],
                    women_cases = [20_000, 18_000, 5_000, 25_000, 9_000, 29_000,
                                   24_000, 0, 21_000, 160_000, 55_000, 97_000],
                    men_deaths = [10_000, 12_000, 13_000, 0, 14_000, 12_000,
                                  16_000, 25_000, 20_000, 500, 25_000, 80_000],
                    men_cases = [30_000, 50_000, 13_000, 0, 16_000, 30_000,
                                 25_000, 220_000, 22_000, 600, 55_000, 115_000],
                    xt_pos = ([150000, 100000, 50000, 0],
                              [0, 50000, 100000, 150000, 200000]),
                    xt_lbls = (['150,000', '100,000', '50,000', 'WOMEN'],
                               ['MEN', '50,000', '100,000', '150,000', '200,000']),
                    year=2007,
                    notes = ('DEATHS', 'NEW CASES')
                   )

def rougier_example(caption_side=0,
                    save_fig=False,
                    save_as=None):
    """
    Wrapper for rougier.plot_paired_categories()
    """
    
    # Data to be represented: cancer data from 2007
    data = rougier_data
    #pp(data)

    # Inputs preparation:
    # -----------------------------------------------------
    W = (data['women_deaths'], data['women_cases'])
    M = (data['men_deaths'], data['men_cases'])

    xt_pos = data['xt_pos']
    xt_lbls = data['xt_lbls']
    yr = data['year']
    notes = data['notes']

    # Caption:
    cases = np.round((np.sum(W[1]) + np.sum(M[1]))/1_000_000, 1)
    caption_hdr = 'Leading Causes\nOf Cancer Deaths'
    caption_txt = 'In {:d}, there were more\nthan {:.1f} million'.format(yr, cases)
    caption_txt += ' new cases\nof cancer in the USA.'
    
    if caption_side == 0:
        caption_args = {} # use defaults
    else:
        caption_args = {'y_pos_pct': 0.4, 'x_pos_offset':55_000}

    fig_style = {'figsize':(14,7), 'facecolor':None}

    plot_paired_categories(data['diseases'],
                           W, M,
                           xt_pos, xt_lbls,
                           face_cols=['red','blue'],
                           equal_lims=True,
                           y_lbl_ofs=0.5,
                           pair_labels=notes,
                           annotate_side='both',
                           caption_ax=caption_side,
                           caption=[caption_hdr, caption_txt],
                           caption_kwargs=caption_args,
                           fig_style=fig_style,
                           save_as=save_as,
                           save_fig=save_fig)
