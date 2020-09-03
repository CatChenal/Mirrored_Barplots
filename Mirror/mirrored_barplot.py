__doc__ = """
Output a bar plot of two paired-series mirrored around
x = 0 (orient='horizontal, default or y = 0 (orient='vertical').
"""

import numpy as np
import matplotlib.pyplot as plt
from functools import partial


def mirrorbar_despine(ax, orient='horizontal'):
    sp = ['left', 'top', 'right', 'bottom']
    
    orient = orient.lower()[0]
    if orient == 'h':
        sp.pop(0)
    else:
        sp.pop(-1)
    for s in sp:
        ax.spines[s].set_visible(False)
        

def mirrored_bar_add_labels(ax, y1, y2,
                            bar_params,
                            orient='horizontal',
                            txt_color=['#ff6347','#74b088'],
                            round_to=2):
    
    font_style = {'fontweight':'bold'}
    
    o = orient.lower()[0]
        
    for s in [0, 1]:      
        ofs = bar_params[o][s]['ofs']
        ha = bar_params[o][s]['ha']
        va = bar_params[o][s]['va']
        col = txt_color[s]

        for b in bar_params[o][s]['b']:
            w = b.get_width()
            h = b.get_height()

            if o == 'v':
                xloc = w + ofs
                yloc = b.get_y() + h / 2
                t = w
            else:
                xloc = b.get_x() + w / 2
                yloc = h + ofs
                t = h

            ax.text(xloc, yloc,
                    '{:.{}f}'.format(np.abs(t), round_to),
                    color=col,
                    ha=ha,
                    va=va,
                    **font_style)

# TODO: retrieve seaborn_args: extra args to seaborn plot not found in plt
def add_mirrored_bars(ax, M, S1, S2,
                      bw,
                      fc, ec,
                      alpha,
                      series_labels,
                      axis_label,
                      orient,
                      round_to,
                      label_bars
                      #use_seaborn=False,seaborn_args=None
                     ):
    offset = 0.02
    bar_params = {'h': {0:{'b':None,
                           'ofs':-offset,
                           'ha':'center',
                           'va':'top'},
                        1:{'b':None,
                           'ofs':offset,
                           'ha':'center',
                           'va':'bottom'}},
                  'v': {0:{'b':None,
                           'ofs':-offset,
                           'ha':'right',
                           'va':'center'},
                        1:{'b':None,
                          'ofs':offset,
                          'ha':'left',
                          'va':'center'}}
                 }
    o = orient.lower()[0]
    b = ax.bar if o == 'h' else ax.barh
    '''
    if not use_seaborn:
        b = ax.bar if o == 'h' else ax.barh
    else:
        if o == 'h':
            b = partial(sns.barplot, {'ax':ax})
        else:
            b = partial(sns.barplot, {'ax':ax, 'orient':'h'})
    '''
    for i in range(2):
        S = -S1 if i == 0 else S2
            
        bar_params[o][i]['b'] = b(M, S,
                                  bw,
                                  label=series_labels[i],
                                  facecolor=fc[i],
                                  edgecolor=ec[i],
                                  alpha=alpha,
                                  align='center'
                                 )
    #ax.figure.canvas.draw()
    
    if o == 'h':
        #ax.autoscale_view(tight=True, scalex=False, scaley=True)
        ax.yaxis.set_major_formatter('{x:.1f}')
        ax.set(xticks=[], ylabel=axis_label)

    else:
        #ax.autoscale_view(tight=True, scalex=True, scaley=False)
        ax.xaxis.set_major_formatter('{x:.1f}')
        ax.set(yticks=[], xlabel=axis_label)
    
    ax.figure.canvas.draw()
        
    if label_bars:
        mirrored_bar_add_labels(ax, S1, S2,
                                bar_params,
                                orient=orient,
                                txt_color=fc,
                                round_to=round_to)
    
    return ax
    

def validate_series(s1, s2, max_ratio=15):

    if len(s1) != len(s2):
        raise ValueError("The two series must have indentical lengths.")
    if np.any(s1 < 0) or np.any(s2 < 0):
        raise ValueError("The two series must be strictly positive.")
        
    rng1 = max(s1) - min(s1)
    rng2 = max(s2) - min(s2)
    if rng1 <= rng2:
        ratio = rng2//rng1
    else:
        ratio = rng1//rng2
    if ratio >= max_ratio:
        msg = F'The series ranges differ by at least {max_ratio}X: '
        msg += 'the visualization may not be optimal.'
        print(msg)
            
        
def mirrored_barplot(ax, M, S1, S2,
                     orient='horizontal',
                     title=None,
                     axis_label='Series values',
                     series_labels=['Series1', 'Series2'],
                     fc=['#ff6347','#74b088'],
                     ec=['w','w'],
                     bw=0.8,
                     alpha=0.75,
                     legend=True,
                     label_bars=False,
                     tc=['#ff6347','#74b088'],
                     round_to=2,
                     style='seaborn',
                     #use_seaborn=False, seaborn_args=None
                    ):
    """Bar plot two paired series reflected around a y=0 or x=0 line.
       Note: The two series values are assumed to have the same order of magnitude.
       Inputs:
       -------
       :param: ax: plot axis
       :param: M, S1, S2: non-negative series
       :param: orient (str): 'h', 'horizontal', 'v', 'vertical';
               If orient in ['h', 'horizontal']: -S1: will mirror S2 around
               the horizontal line y=0 else, -S1 will mirror S2 around the vertical line x=0
       :param: axis_label, series_labels: 2-tuple of str
       :param: fc, ec: 2-tuples of str for facecolor and edge color, respectively
       :param: bw: bar width (default=0.8)
       :param: alpha (0,1): bar color transparency
       :param: legend: (bool): show a legend if True (default=True)
       :param: label_bars (bool): label each bar if True (default=False)
       :param: tc: 2-tuple of color str for text color
       :param: round_to (int): precision of numbers
       :param: style (str): to set pyplot.style.context (default='seaborn');
       
       Call example:
       -------------
       fig, ax = plt.subplots(1, figsize=(6,4))
       
       mirrored_barplot(ax, x, y1, y2,
                        title="My mirrored pair barplot.");
       
    """
    validate_series(S1, S2)
    
    if (len(fc) != 2) or (len(ec) != 2) or (len(tc) != 2):
        msg = "facecolor (fc), edgecolor (ec) and "
        msg += "text_color (tc) parameters must be 2-tuples."
        raise ValueError(msg)

    orient = orient.lower()[0]
    if orient not in ['h', 'v']:
        raise ValueError("orient not in ['h', 'horizontal', 'v', 'vertical'].")
        
    with plt.style.context(style=style):
        ax = add_mirrored_bars(ax, M, S1, S2,
                               bw=bw,
                               fc=fc, ec=ec,
                               alpha=alpha, 
                               series_labels=series_labels,
                               axis_label=axis_label,
                               orient=orient,
                               round_to=round_to,
                               label_bars=label_bars
                               #use_seaborn=False, seaborn_args=None
                              )
        
        mirrorbar_despine(ax, orient=orient)
        
        legend_cols = 2
        
        if orient == 'h':
            grid_axis = 'y'
            y = 1.
        else:
            grid_axis = 'x'
            y = .98

        ax.grid(which='major',
                axis=grid_axis,
                color='w',
                linewidth=0.7)
            
        if legend:
            ax.legend(bbox_to_anchor=(0.2, y, 1., .10), 
                      loc='lower left',
                      ncol=legend_cols, 
                      frameon=False,
                      borderaxespad=0.)

        if title:
            ax.set_title(title, y=1.05)
        
        plt.tight_layout()
        

def sample_data(n=10, seed=None, fixed=1, y2_factor=4):
    '''
    :fixed=1: hard coded series, else random.
    '''
    if fixed:
        x = np.arange(10)
        y1 = np.array([1.4359949 , 0.92333361, 1.23972998, 1.00472567, 0.85222068,
                       0.66516741, 0.48185945, 0.25993093, 0.48578129, 0.12668273])
        y2 = np.array([0.81056692, 0.68811394, 0.45383198, 0.52975234, 0.35533196,
                       0.44633379, 0.37079506, 0.22413553, 0.18465615, 0.5398227])
    else: 
        x = np.arange(n)
        if seed is not None:
            np.random.seed(seed)
        y1 = np.random.random_sample(n)
        y2 = np.random.random_sample(n)*y2_factor
    
    return x, y1, y2
