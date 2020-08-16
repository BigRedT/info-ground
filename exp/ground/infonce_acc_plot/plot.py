import os
import numpy as np
import matplotlib.pyplot as plt

import utils.io as io
from global_constants import misc_paths

def get_infonce_data(infonce_dir,layers):
    infonce_data = io.load_json_object(
        os.path.join(
            infonce_dir,
            f'infonce_{layers}_layer.json'))
    iters = []
    losses = []
    for time,it,loss in infonce_data:
        if it==0:
            continue

        iters.append(it)
        losses.append(round(loss,2))

    return iters, losses


def get_acc_data(acc_dir,iters):
    accs = [None]*len(iters)
    for i,it in enumerate(iters):
        results_json = os.path.join(acc_dir,f'results_val_{it}.json')
        
        if not os.path.exists(results_json):
            continue
            
        accs[i] = io.load_json_object(results_json)['pt_recall']
    
    return accs


def create_point_label(x,y,label,color,markersize,marker):
    plt.plot(x,y,c=color,markersize=markersize,marker=marker)
    plt.annotate(label,(x+0.025,y),c=color,va='center',fontsize=9,family='serif')

def main():
    infonce_dir = os.path.join(
        os.getcwd(),
        'exp/pretrain_coco_noun_negs/infonce_acc_plot')   
    exp_dir = '/shared/rsaas/tgupta6/Data/context-regions/coco_exp'

    colors = ['r','g','b']
    num_layers = [1,2,3]
    infonce_losses = {}
    handles = [None]*3
    labels = ['Linear', 'MLP w/ 1 hidden layer', 'MLP w/ 2 hidden layers']
    arrowcolor='k' #(0.3,0.3,0.3)
    ha = ['right','left','right']
    for i,l in enumerate(num_layers):
        iters,losses = get_infonce_data(infonce_dir,l)

        acc_dir = os.path.join(
            exp_dir,
            f'loss_wts_neg_noun_1_self_sup_1_lang_sup_1_no_context_vgdet_nonlinear_infonce_{l}_layer_adj_batch_50')
        accs = get_acc_data(acc_dir,iters)
        
        bounds = [np.log(50)-infonce for infonce in losses]
        handles[i], = plt.plot(bounds,accs,c=colors[i],markersize=0,marker='o',linewidth=1.5,label=labels[i])

        k = np.argmax(accs)
        labels.append(iters[k])
        plt.annotate(
            str(iters[k]//1000) + 'K Iters',
            c=arrowcolor,
            xy=(bounds[k],accs[k]),
            xytext=(3.35,accs[k]),
            fontsize=9,
            family='serif',
            arrowprops=dict(arrowstyle="->",linestyle='-',ec=arrowcolor,fc=arrowcolor),
            va='center')
        
        plt.plot(bounds[0],accs[0],c=colors[i],markersize=4,marker='o')
        plt.plot(bounds[k],accs[k],c=colors[i],markersize=6,marker='*')
        plt.plot(bounds[-1],accs[-1],c=colors[i],markersize=4,marker='s')
        
    # Manual legend for iterations
    lx = 3.04 #49.45
    ly = 73 #66
    d = 0.8
    #plt.annotate('Iterations:',(lx-0.005,ly),c=arrowcolor,va='center',fontsize=9,family='serif',weight='bold')
    create_point_label(lx,ly,'4K Iters',arrowcolor,markersize=4,marker='o')
    create_point_label(lx,ly-d,'80K Iters',arrowcolor,markersize=4,marker='s')
    create_point_label(lx,ly-2*d,'Best Accuracy',arrowcolor,markersize=6,marker='*')

    # Legend for layers
    plt.plot()
    plt.legend(
        handles=handles,
        loc='upper left',
        frameon=False,
        prop={'size':9,'family':'serif'})
    plt.xlabel("InfoNCE lower bound on COCO (Val)",fontsize=9,family='serif')
    plt.ylabel('Pointing accuracy on Flickr30k Entities (Val)',fontsize=9,family='serif')

    plt.yticks(size=9,family='serif')
    plt.xticks(size=9,family='serif')
    # a = plt.gca()
    # import pdb; pdb.set_trace()
    # a.set_xticklabels(a.get_xticks(), {'family':'serif'})
    # a.set_yticklabels(a.get_yticks(), {'family':'serif'})

    figname = os.path.join(misc_paths['scratch_dir'],'infonce_acc_plot.png')
    plt.savefig(figname,dpi=600,bbox_inches='tight')


if __name__=='__main__':
    main()