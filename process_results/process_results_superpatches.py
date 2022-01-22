import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as CM

import numpy as np;
import pickle;
import os;
import sys;
#import matplotlib.pyplot as plt; 
import seaborn as sns



def process_results_hist(in_dir, out_dir, model_prefix, dataset_name, threshold):

    result_files_prefix = os.path.join(in_dir, model_prefix);
    out_files_prefix = os.path.join(in_dir, model_prefix);
    lbl = np.load(result_files_prefix + '_individual_labels.npy', allow_pickle=True);
    pred = np.load(result_files_prefix + '_pred_new.npy', allow_pickle=True);

    # label values are: 1, 2, 3, 4,; empty,  ignore value of 4 and empty
    lbl[np.where(l==4)] = 0   ;
    # to get the average score label need to get the count of scores available for each super patch
    b = lbl>0;
    n = b.sum(axis = 1);
    n[np.where(n ==0)] = -1; # set to -1 the count = 0 to avoid division by zero
    # get the average score label by summing each patch scores and divide by count then round
    lbl2 = np.divide(lbl.sum(axis = 1), n);
    lbl2 = np.round(lbl2);

    # get the sub patches that are predicted positive according to threshold
    # the pred is super patch -> sub patch -> logit neg, logit pos
    pred= pred.squeeze();
    pred = pred[:,:,1] ;
    pred_b = pred > threshold ;

    # get the number of subpatches predicted positive in each superpatch
    pred_n = pred_b.sum(axis = 1)
    # get the number of subpatches predicted positive in each superpatch in each score label category 1,2,3
    pred_n1 = pred_n[np.where(lbl2 == 1)]
    pred_n2 = pred_n[np.where(lbl2 == 2)]
    pred_n3 = pred_n[np.where(lbl2 == 3)]

    # Calculate the histogram of the pos count in each label category
    hist1 = np.histogram(pred_n1, bins=np.arange(0,65, 5))
    hist1[0].dump(out_files_prefix + '_' + dataset_name + '_hist1y_step5.npy');
    hist1[1].dump(out_files_prefix + '_' + dataset_name + '_hist1x_step5.npy');
    hist2 = np.histogram(pred_n2, bins=np.arange(0,65, 5))
    hist2[0].dump(out_files_prefix + '_' + dataset_name + '_hist2y_step5.npy');
    hist2[1].dump(out_files_prefix + '_' + dataset_name + '_hist2x_step5.npy');
    hist3 = np.histogram(pred_n3, bins=np.arange(0,65, 5))
    hist3[0].dump(out_files_prefix + '_' + dataset_name + '_hist3y_step5.npy');
    hist3[1].dump(out_files_prefix + '_' + dataset_name + '_hist3x_step5.npy');

    # Visualize the histograms
    for i in range(1,4):
        histy = np.load(out_files_prefix + '_' + dataset_name + '_hist'+str(i)+'y_step5.npy', allow_pickle=True)
        histx = np.load(out_files_prefix + '_' + dataset_name + '_hist'+str(i)+'x_step5.npy', allow_pickle=True)
        plt.bar(histx_s5[1:], histy_s5)
        #plt.plot(histx_s5[1:], histy_s5, label="inc")
        plt.plot(histx_s5[1:], histy_s5)
        plt.legend()
        plt.xticks(np.arange(0,histx_s5[-1]+1,5))
        plt.show();

    return;

def process_le_results_violin_use_anno_csv_outraw_merged(out_dir, anno_filepath_csv, dataset_name, title_line=None):

    #result_files_prefix = os.path.join(in_dir, model_prefix );

    anno_arr = np.loadtxt(anno_filepath_csv, delimiter=',', dtype=str)
    anno_arr = np.delete(anno_arr , 0, axis=0)
    anno_ctype = anno_arr[:, 0] 
    anno_filepath = anno_arr[:, 1:3] 
    anno_lbl = anno_arr[:, 3:-1]
    anno_arr_lbl_full = anno_arr[:, 3:-1]
    print('anno_lbl',anno_lbl.shape)
    anno_lbl[anno_lbl=='']='0'
    anno_lbl[anno_lbl==' ']='0'
    anno_lbl = anno_lbl.astype(int)
    anno_lbl[np.where(anno_lbl==4)] = 0   ;
    # to get the average score label need to get the count of scores available for each super patch
    b = anno_lbl>0;
    n = b.sum(axis = 1);
    n[np.where(n ==0)] = -1; # set to -1 the count = 0 to avoid division by zero
    # get the average score label by summing each patch scores and divide by count then round
    anno_lbl2 = np.divide(anno_lbl.sum(axis = 1), n);
    anno_lbl2 = np.round(anno_lbl2);
    ctypes = np.unique(anno_ctype)
    pred_patch_count = anno_arr[:,-1]

    pred_patch_count_n1 = pred_patch_count[np.where(anno_lbl2== 1)]
    pred_patch_count_n2 = pred_patch_count[np.where(anno_lbl2== 2)]
    pred_patch_count_n3 = pred_patch_count[np.where(anno_lbl2== 3)]


    fig,ax = plt.subplots(1)
    sns.set(style="whitegrid")
    ax = sns.violinplot(data=[pred_patch_count_n1,pred_patch_count_n2,pred_patch_count_n3], cut=0, ax=ax, palette='tab10')
    ax.set(xticklabels=['low', 'medium', 'high'])
    fig.savefig(os.path.join(out_dir, 'violin'+'_patch_count_'+'all'+'.png'));

    fig,ax = plt.subplots(1)
    sns.set(style="whitegrid")
    ax = sns.violinplot(data=[pred_patch_count_n1,pred_patch_count_n2,pred_patch_count_n3], cut=0, ax=ax, palette='tab10', scale='width')
    ax.set(xticklabels=['low', 'medium', 'high'])
    fig.savefig(os.path.join(out_dir, 'violin'+'_patch_count_'+'all'+'_samewidth'+'.png'));

    if(title_line is None):
        out_filepath_all = os.path.join(out_dir, 'superpatches' + '_lbl_n_pred_'+'all'+'_full_label.txt')
    else:
        out_filepath_all = os.path.join(out_dir, 'superpatches' + '_lbl_n_pred_'+'all'+'_full_label_wtitle.txt')
    if((not(title_line is None)) and (not os.path.exists(out_filepath_all))):
        with open(out_filepath_all, 'a') as file:
            file.write(title_line + '\n')

    ctypes = np.unique(anno_ctype)
    for ctype in ctypes:
        anno_lbl2_ctype = anno_lbl2[np.where(anno_ctype == ctype)]
        anno_arr_lbl_full_ctype = anno_arr_lbl_full[np.where(anno_ctype == ctype)]
        #pred_cell_count_ctype = pred_cell_count[np.where(anno_ctype == ctype)]
        pred_patch_count_ctype = pred_patch_count[np.where(anno_ctype == ctype)]
        anno_arr_ctype = anno_arr[np.where(anno_ctype == ctype)]
    
        #pred_cell_count_n1 = pred_cell_count_ctype[np.where(anno_lbl2_ctype == 1)]
        #pred_cell_count_n2 = pred_cell_count_ctype[np.where(anno_lbl2_ctype == 2)]
        #pred_cell_count_n3 = pred_cell_count_ctype[np.where(anno_lbl2_ctype == 3)]

        pred_patch_count_n1 = pred_patch_count_ctype[np.where(anno_lbl2_ctype == 1)]
        pred_patch_count_n2 = pred_patch_count_ctype[np.where(anno_lbl2_ctype == 2)]
        pred_patch_count_n3 = pred_patch_count_ctype[np.where(anno_lbl2_ctype == 3)]

        ##fig,ax = plt.subplots(1)
        ##sns.set(style="whitegrid")
        ##ax = sns.violinplot(data=[pred_cell_count_n1,pred_cell_count_n2,pred_cell_count_n3], cut=0, ax=ax, palette='tab10')
        ##ax.set(xticklabels=['low', 'medium', 'high'])
        ##fig.savefig(os.path.join(out_dir, 'violin'+'_cell_count_'+ctype+'.png'));

        #fig,ax = plt.subplots(1)
        #sns.set(style="whitegrid")
        #ax = sns.violinplot(data=[pred_cell_count_n1,pred_cell_count_n2,pred_cell_count_n3], cut=0, ax=ax, palette='tab10', scale='width')
        #ax.set(xticklabels=['low', 'medium', 'high'])
        #fig.savefig(os.path.join(out_dir, 'violin'+'_cell_count_'+ctype+'_samewidth'+'.png'));

        fig,ax = plt.subplots(1)
        sns.set(style="whitegrid")
        ax = sns.violinplot(data=[pred_patch_count_n1,pred_patch_count_n2,pred_patch_count_n3], cut=0, ax=ax, palette='tab10')
        ax.set(xticklabels=['low', 'medium', 'high'])
        fig.savefig(os.path.join(out_dir, 'violin'+'_patch_count_'+ctype+'.png'));

        fig,ax = plt.subplots(1)
        sns.set(style="whitegrid")
        ax = sns.violinplot(data=[pred_patch_count_n1,pred_patch_count_n2,pred_patch_count_n3], cut=0, ax=ax, palette='tab10', scale='width')
        ax.set(xticklabels=['low', 'medium', 'high'])
        fig.savefig(os.path.join(out_dir, 'violin'+'_patch_count_'+ctype+'_samewidth'+'.png'));
    
        if(title_line is None):
            out_filepath = os.path.join(out_dir, 'superpatches' + '_lbl_n_pred_'+ctype+'_full_label.txt')
        else:
            out_filepath = os.path.join(out_dir, 'superpatches' + '_lbl_n_pred_'+ctype+'_full_label_wtitle.txt')
        with open(out_filepath, 'w') as file:
            if(not(title_line is None)):
                file.write(title_line + '\n')
            for i in range(anno_arr_ctype.shape[0]):
                #file.write(anno_arr_ctype[i,1]+'_'+anno_arr_ctype[i,2] + ','+str(int(anno_lbl2_ctype[i]))+ ',' + ctype + ',' + str(pred_cell_count_ctype[i])  + ',' + str(pred_patch_count_ctype[i])   +'\n') ;
                #file.write(anno_arr_ctype[i,1]+'_'+anno_arr_ctype[i,2] + ','+str(int(anno_lbl2_ctype[i]))+ ',' + ctype + ',' + str(pred_cell_count_ctype[i])  + ',' + str(pred_patch_count_ctype[i])   ) ;
                file.write(anno_arr_ctype[i,1]+'_'+anno_arr_ctype[i,2] + ','+str(int(anno_lbl2_ctype[i]))+ ',' + ctype + ',' + str(pred_patch_count_ctype[i])   ) ;
                for j in range(anno_arr_lbl_full_ctype.shape[-1]):
                    file.write(',' + anno_arr_lbl_full_ctype[i,j]) ;            
                file.write('\n') ;

        # all ctypes together in one file
        with open(out_filepath_all, 'a') as file:
            for i in range(anno_arr_ctype.shape[0]):
                #file.write(anno_arr_ctype[i,1]+'_'+anno_arr_ctype[i,2] + ','+str(int(anno_lbl2_ctype[i]))+ ',' + ctype + ',' + str(pred_cell_count_ctype[i])  + ',' + str(pred_patch_count_ctype[i])   +'\n') ;
                #file.write(anno_arr_ctype[i,1]+'_'+anno_arr_ctype[i,2] + ','+str(int(anno_lbl2_ctype[i]))+ ',' + ctype + ',' + str(pred_cell_count_ctype[i])  + ',' + str(pred_patch_count_ctype[i])   ) ;
                file.write(anno_arr_ctype[i,1]+'_'+anno_arr_ctype[i,2] + ','+str(int(anno_lbl2_ctype[i]))+ ',' + ctype  + ',' + str(pred_patch_count_ctype[i])   ) ;
                for j in range(anno_arr_lbl_full_ctype.shape[-1]):
                    file.write(',' + anno_arr_lbl_full_ctype[i,j]) ;            
                file.write('\n') ;


def process_results_violin_use_anno_csv_outraw_merged(in_dir, out_dir, anno_filepath_csv, model_prefix, dataset_name, threshold, title_line=None):

    result_files_prefix = os.path.join(in_dir, model_prefix );

    anno_arr = np.loadtxt(anno_filepath_csv, delimiter=',', dtype=str)
    anno_arr = np.delete(anno_arr , 0, axis=0)
    anno_ctype = anno_arr[:, 0] 
    anno_filepath = anno_arr[:, 1:3] 
    anno_lbl = anno_arr[:, 3:-1]
    anno_arr_lbl_full = anno_arr[:, 3:-1]
    print('anno_lbl',anno_lbl.shape)
    anno_lbl[anno_lbl=='']='0'
    anno_lbl[anno_lbl==' ']='0'
    anno_lbl = anno_lbl.astype(int)
    anno_lbl[np.where(anno_lbl==4)] = 0   ;
    # to get the average score label need to get the count of scores available for each super patch
    b = anno_lbl>0;
    n = b.sum(axis = 1);
    n[np.where(n ==0)] = -1; # set to -1 the count = 0 to avoid division by zero
    # get the average score label by summing each patch scores and divide by count then round
    anno_lbl2 = np.divide(anno_lbl.sum(axis = 1), n);
    anno_lbl2 = np.round(anno_lbl2);
    ctypes = np.unique(anno_ctype)

    #pred = np.load(result_files_prefix + '_pred_new.npy');
    if(os.path.isfile(result_files_prefix + '_pred_new.npy')):
        pred = np.load(result_files_prefix + '_pred_new.npy', allow_pickle=True);
    elif(os.path.isfile(result_files_prefix + '_pred_prob.npy')):
        pred = np.load(result_files_prefix + '_pred_prob.npy', allow_pickle=True);    
    filenames = np.array(pickle.load(open(result_files_prefix + '_filename.pkl', 'rb')));    
    print(filenames, filenames.shape)
    #print('pred.shape = ', pred.shape)
    pred= pred.squeeze();
    print('pred.shape = ', pred.shape)
    #pred = pred[:,:,1] ;
    if(len(pred.shape) > 2 and pred.shape[2]>1):
        pred = pred[:,:,1];
    elif(len(pred.shape) > 2 and pred.shape[2]==1):
        pred = pred[:,:,0];
    elif(len(pred.shape) == 2):
        pred = pred;
    pred_b = pred > threshold ;
    # get the number of subpatches predicted positive in each superpatch
    pred_n = pred_b.sum(axis = 1)

    pred_patch_count = np.zeros(anno_lbl2.shape)
    for i in range(anno_arr.shape[0]):
        filename = anno_filepath[i,0] + '_' + anno_filepath[i,1] + '.png'
        print(filename)
        print(pred_n[filenames==filename]) 
        pred_patch_count[i] = pred_n[filenames==filename]
 
    pred_patch_count_n1 = pred_patch_count[np.where(anno_lbl2== 1)]
    pred_patch_count_n2 = pred_patch_count[np.where(anno_lbl2== 2)]
    pred_patch_count_n3 = pred_patch_count[np.where(anno_lbl2== 3)]


    fig,ax = plt.subplots(1)
    sns.set(style="whitegrid")
    ax = sns.violinplot(data=[pred_patch_count_n1,pred_patch_count_n2,pred_patch_count_n3], cut=0, ax=ax, palette='tab10')
    ax.set(xticklabels=['low', 'medium', 'high'])
    fig.savefig(os.path.join(out_dir, 'violin'+'_patch_count_'+'all'+'.png'));

    fig,ax = plt.subplots(1)
    sns.set(style="whitegrid")
    ax = sns.violinplot(data=[pred_patch_count_n1,pred_patch_count_n2,pred_patch_count_n3], cut=0, ax=ax, palette='tab10', scale='width')
    ax.set(xticklabels=['low', 'medium', 'high'])
    fig.savefig(os.path.join(out_dir, 'violin'+'_patch_count_'+'all'+'_samewidth'+'.png'));

    if(title_line is None):
        out_filepath_all = os.path.join(out_dir, 'superpatches' + '_lbl_n_pred_'+'all'+'_full_label.txt')
    else:
        out_filepath_all = os.path.join(out_dir, 'superpatches' + '_lbl_n_pred_'+'all'+'_full_label_wtitle.txt')
    if((not(title_line is None)) and (not os.path.exists(out_filepath_all))):
        with open(out_filepath_all, 'a') as file:
            file.write(title_line + '\n')

    ctypes = np.unique(anno_ctype)
    for ctype in ctypes:
        anno_lbl2_ctype = anno_lbl2[np.where(anno_ctype == ctype)]
        anno_arr_lbl_full_ctype = anno_arr_lbl_full[np.where(anno_ctype == ctype)]
        #pred_cell_count_ctype = pred_cell_count[np.where(anno_ctype == ctype)]
        pred_patch_count_ctype = pred_patch_count[np.where(anno_ctype == ctype)]
        anno_arr_ctype = anno_arr[np.where(anno_ctype == ctype)]
    
        #pred_cell_count_n1 = pred_cell_count_ctype[np.where(anno_lbl2_ctype == 1)]
        #pred_cell_count_n2 = pred_cell_count_ctype[np.where(anno_lbl2_ctype == 2)]
        #pred_cell_count_n3 = pred_cell_count_ctype[np.where(anno_lbl2_ctype == 3)]

        pred_patch_count_n1 = pred_patch_count_ctype[np.where(anno_lbl2_ctype == 1)]
        pred_patch_count_n2 = pred_patch_count_ctype[np.where(anno_lbl2_ctype == 2)]
        pred_patch_count_n3 = pred_patch_count_ctype[np.where(anno_lbl2_ctype == 3)]

        ##fig,ax = plt.subplots(1)
        ##sns.set(style="whitegrid")
        ##ax = sns.violinplot(data=[pred_cell_count_n1,pred_cell_count_n2,pred_cell_count_n3], cut=0, ax=ax, palette='tab10')
        ##ax.set(xticklabels=['low', 'medium', 'high'])
        ##fig.savefig(os.path.join(out_dir, 'violin'+'_cell_count_'+ctype+'.png'));

        #fig,ax = plt.subplots(1)
        #sns.set(style="whitegrid")
        #ax = sns.violinplot(data=[pred_cell_count_n1,pred_cell_count_n2,pred_cell_count_n3], cut=0, ax=ax, palette='tab10', scale='width')
        #ax.set(xticklabels=['low', 'medium', 'high'])
        #fig.savefig(os.path.join(out_dir, 'violin'+'_cell_count_'+ctype+'_samewidth'+'.png'));

        fig,ax = plt.subplots(1)
        sns.set(style="whitegrid")
        ax = sns.violinplot(data=[pred_patch_count_n1,pred_patch_count_n2,pred_patch_count_n3], cut=0, ax=ax, palette='tab10')
        ax.set(xticklabels=['low', 'medium', 'high'])
        fig.savefig(os.path.join(out_dir, 'violin'+'_patch_count_'+ctype+'.png'));

        fig,ax = plt.subplots(1)
        sns.set(style="whitegrid")
        ax = sns.violinplot(data=[pred_patch_count_n1,pred_patch_count_n2,pred_patch_count_n3], cut=0, ax=ax, palette='tab10', scale='width')
        ax.set(xticklabels=['low', 'medium', 'high'])
        fig.savefig(os.path.join(out_dir, 'violin'+'_patch_count_'+ctype+'_samewidth'+'.png'));
    
        if(title_line is None):
            out_filepath = os.path.join(out_dir, 'superpatches' + '_lbl_n_pred_'+ctype+'_full_label.txt')
        else:
            out_filepath = os.path.join(out_dir, 'superpatches' + '_lbl_n_pred_'+ctype+'_full_label_wtitle.txt')
        with open(out_filepath, 'w') as file:
            if(not(title_line is None)):
                file.write(title_line + '\n')
            for i in range(anno_arr_ctype.shape[0]):
                #file.write(anno_arr_ctype[i,1]+'_'+anno_arr_ctype[i,2] + ','+str(int(anno_lbl2_ctype[i]))+ ',' + ctype + ',' + str(pred_cell_count_ctype[i])  + ',' + str(pred_patch_count_ctype[i])   +'\n') ;
                #file.write(anno_arr_ctype[i,1]+'_'+anno_arr_ctype[i,2] + ','+str(int(anno_lbl2_ctype[i]))+ ',' + ctype + ',' + str(pred_cell_count_ctype[i])  + ',' + str(pred_patch_count_ctype[i])   ) ;
                file.write(anno_arr_ctype[i,1]+'_'+anno_arr_ctype[i,2] + ','+str(int(anno_lbl2_ctype[i]))+ ',' + ctype + ',' + str(pred_patch_count_ctype[i])   ) ;
                for j in range(anno_arr_lbl_full_ctype.shape[-1]):
                    file.write(',' + anno_arr_lbl_full_ctype[i,j]) ;            
                file.write('\n') ;

        # all ctypes together in one file
        with open(out_filepath_all, 'a') as file:
            for i in range(anno_arr_ctype.shape[0]):
                #file.write(anno_arr_ctype[i,1]+'_'+anno_arr_ctype[i,2] + ','+str(int(anno_lbl2_ctype[i]))+ ',' + ctype + ',' + str(pred_cell_count_ctype[i])  + ',' + str(pred_patch_count_ctype[i])   +'\n') ;
                #file.write(anno_arr_ctype[i,1]+'_'+anno_arr_ctype[i,2] + ','+str(int(anno_lbl2_ctype[i]))+ ',' + ctype + ',' + str(pred_cell_count_ctype[i])  + ',' + str(pred_patch_count_ctype[i])   ) ;
                file.write(anno_arr_ctype[i,1]+'_'+anno_arr_ctype[i,2] + ','+str(int(anno_lbl2_ctype[i]))+ ',' + ctype  + ',' + str(pred_patch_count_ctype[i])   ) ;
                for j in range(anno_arr_lbl_full_ctype.shape[-1]):
                    file.write(',' + anno_arr_lbl_full_ctype[i,j]) ;            
                file.write('\n') ;

def process_results_violin_use_anno_csv_outraw_reviewed(in_dir, out_dir, anno_filepath_csv, model_prefix, dataset_name, threshold, title_line=None):

    result_files_prefix = os.path.join(in_dir, model_prefix );

    anno_arr = np.loadtxt(anno_filepath_csv, delimiter=',', dtype=str)
    anno_arr = np.delete(anno_arr , 0, axis=0)
    anno_ctype = anno_arr[:, 0] 
    anno_filepath = anno_arr[:, 1] 
    anno_lbl = anno_arr[:, 2:5]
    anno_arr_lbl_full = anno_arr[:, 2:5]
    print('anno_lbl',anno_lbl.shape)
    anno_lbl[anno_lbl=='']='0'
    anno_lbl[anno_lbl==' ']='0'
    anno_lbl = anno_lbl.astype(int)
    anno_lbl[np.where(anno_lbl==4)] = 0   ;
    # to get the average score label need to get the count of scores available for each super patch
    b = anno_lbl>0;
    n = b.sum(axis = 1);
    n[np.where(n ==0)] = -1; # set to -1 the count = 0 to avoid division by zero
    # get the average score label by summing each patch scores and divide by count then round
    anno_lbl2 = np.divide(anno_lbl.sum(axis = 1), n);
    anno_lbl2 = np.round(anno_lbl2);
    ctypes = np.unique(anno_ctype)

    #pred = np.load(result_files_prefix + '_pred_new.npy');
    if(os.path.isfile(result_files_prefix + '_pred_new.npy')):
        pred = np.load(result_files_prefix + '_pred_new.npy', allow_pickle=True);
    elif(os.path.isfile(result_files_prefix + '_pred_prob.npy')):
        pred = np.load(result_files_prefix + '_pred_prob.npy', allow_pickle=True);    
    filenames = np.array(pickle.load(open(result_files_prefix + '_filename.pkl', 'rb')));    
    print(filenames, filenames.shape)
    #print('pred.shape = ', pred.shape)
    pred= pred.squeeze();
    print('pred.shape = ', pred.shape)
    #pred = pred[:,:,1] ;
    if(len(pred.shape) > 2 and pred.shape[2]>1):
        pred = pred[:,:,1];
    elif(len(pred.shape) > 2 and pred.shape[2]==1):
        pred = pred[:,:,0];
    elif(len(pred.shape) == 2):
        pred = pred;
    pred_b = pred > threshold ;
    # get the number of subpatches predicted positive in each superpatch
    pred_n = pred_b.sum(axis = 1)

    pred_patch_count = np.zeros(anno_lbl2.shape)
    for i in range(anno_arr.shape[0]):
        filename = anno_filepath[i] 
        print(filename)
        print(pred_n[filenames==filename]) 
        pred_patch_count[i] = pred_n[filenames==filename]
 
    pred_patch_count_n1 = pred_patch_count[np.where(anno_lbl2== 1)]
    pred_patch_count_n2 = pred_patch_count[np.where(anno_lbl2== 2)]
    pred_patch_count_n3 = pred_patch_count[np.where(anno_lbl2== 3)]


    fig,ax = plt.subplots(1)
    sns.set(style="whitegrid")
    ax = sns.violinplot(data=[pred_patch_count_n1,pred_patch_count_n2,pred_patch_count_n3], cut=0, ax=ax, palette='tab10')
    ax.set(xticklabels=['low', 'medium', 'high'])
    fig.savefig(os.path.join(out_dir, 'violin'+'_patch_count_'+'all'+'.png'));

    fig,ax = plt.subplots(1)
    sns.set(style="whitegrid")
    ax = sns.violinplot(data=[pred_patch_count_n1,pred_patch_count_n2,pred_patch_count_n3], cut=0, ax=ax, palette='tab10', scale='width')
    ax.set(xticklabels=['low', 'medium', 'high'])
    fig.savefig(os.path.join(out_dir, 'violin'+'_patch_count_'+'all'+'_samewidth'+'.png'));

    if(title_line is None):
        out_filepath_all = os.path.join(out_dir, 'superpatches' + '_lbl_n_pred_'+'all'+'_full_label.txt')
    else:
        out_filepath_all = os.path.join(out_dir, 'superpatches' + '_lbl_n_pred_'+'all'+'_full_label_wtitle.txt')
    if((not(title_line is None)) and (not os.path.exists(out_filepath_all))):
        with open(out_filepath_all, 'a') as file:
            file.write(title_line + '\n')

    ctypes = np.unique(anno_ctype)
    for ctype in ctypes:
        anno_lbl2_ctype = anno_lbl2[np.where(anno_ctype == ctype)]
        anno_arr_lbl_full_ctype = anno_arr_lbl_full[np.where(anno_ctype == ctype)]
        #pred_cell_count_ctype = pred_cell_count[np.where(anno_ctype == ctype)]
        pred_patch_count_ctype = pred_patch_count[np.where(anno_ctype == ctype)]
        anno_arr_ctype = anno_arr[np.where(anno_ctype == ctype)]
    
        #pred_cell_count_n1 = pred_cell_count_ctype[np.where(anno_lbl2_ctype == 1)]
        #pred_cell_count_n2 = pred_cell_count_ctype[np.where(anno_lbl2_ctype == 2)]
        #pred_cell_count_n3 = pred_cell_count_ctype[np.where(anno_lbl2_ctype == 3)]

        pred_patch_count_n1 = pred_patch_count_ctype[np.where(anno_lbl2_ctype == 1)]
        pred_patch_count_n2 = pred_patch_count_ctype[np.where(anno_lbl2_ctype == 2)]
        pred_patch_count_n3 = pred_patch_count_ctype[np.where(anno_lbl2_ctype == 3)]

        ##fig,ax = plt.subplots(1)
        ##sns.set(style="whitegrid")
        ##ax = sns.violinplot(data=[pred_cell_count_n1,pred_cell_count_n2,pred_cell_count_n3], cut=0, ax=ax, palette='tab10')
        ##ax.set(xticklabels=['low', 'medium', 'high'])
        ##fig.savefig(os.path.join(out_dir, 'violin'+'_cell_count_'+ctype+'.png'));

        #fig,ax = plt.subplots(1)
        #sns.set(style="whitegrid")
        #ax = sns.violinplot(data=[pred_cell_count_n1,pred_cell_count_n2,pred_cell_count_n3], cut=0, ax=ax, palette='tab10', scale='width')
        #ax.set(xticklabels=['low', 'medium', 'high'])
        #fig.savefig(os.path.join(out_dir, 'violin'+'_cell_count_'+ctype+'_samewidth'+'.png'));

        fig,ax = plt.subplots(1)
        sns.set(style="whitegrid")
        ax = sns.violinplot(data=[pred_patch_count_n1,pred_patch_count_n2,pred_patch_count_n3], cut=0, ax=ax, palette='tab10')
        ax.set(xticklabels=['low', 'medium', 'high'])
        fig.savefig(os.path.join(out_dir, 'violin'+'_patch_count_'+ctype+'.png'));

        fig,ax = plt.subplots(1)
        sns.set(style="whitegrid")
        ax = sns.violinplot(data=[pred_patch_count_n1,pred_patch_count_n2,pred_patch_count_n3], cut=0, ax=ax, palette='tab10', scale='width')
        ax.set(xticklabels=['low', 'medium', 'high'])
        fig.savefig(os.path.join(out_dir, 'violin'+'_patch_count_'+ctype+'_samewidth'+'.png'));
    
        if(title_line is None):
            out_filepath = os.path.join(out_dir, 'superpatches' + '_lbl_n_pred_'+ctype+'_full_label.txt')
        else:
            out_filepath = os.path.join(out_dir, 'superpatches' + '_lbl_n_pred_'+ctype+'_full_label_wtitle.txt')
        with open(out_filepath, 'w') as file:
            if(not(title_line is None)):
                file.write(title_line + '\n')
            for i in range(anno_arr_ctype.shape[0]):
                #file.write(anno_arr_ctype[i,1]+'_'+anno_arr_ctype[i,2] + ','+str(int(anno_lbl2_ctype[i]))+ ',' + ctype + ',' + str(pred_cell_count_ctype[i])  + ',' + str(pred_patch_count_ctype[i])   +'\n') ;
                #file.write(anno_arr_ctype[i,1]+'_'+anno_arr_ctype[i,2] + ','+str(int(anno_lbl2_ctype[i]))+ ',' + ctype + ',' + str(pred_cell_count_ctype[i])  + ',' + str(pred_patch_count_ctype[i])   ) ;
                file.write(anno_arr_ctype[i,1]+'_'+anno_arr_ctype[i,2] + ','+str(int(anno_lbl2_ctype[i]))+ ',' + ctype + ',' + str(pred_patch_count_ctype[i])   ) ;
                for j in range(anno_arr_lbl_full_ctype.shape[-1]):
                    file.write(',' + anno_arr_lbl_full_ctype[i,j]) ;            
                file.write('\n') ;

        # all ctypes together in one file
        with open(out_filepath_all, 'a') as file:
            for i in range(anno_arr_ctype.shape[0]):
                #file.write(anno_arr_ctype[i,1]+'_'+anno_arr_ctype[i,2] + ','+str(int(anno_lbl2_ctype[i]))+ ',' + ctype + ',' + str(pred_cell_count_ctype[i])  + ',' + str(pred_patch_count_ctype[i])   +'\n') ;
                #file.write(anno_arr_ctype[i,1]+'_'+anno_arr_ctype[i,2] + ','+str(int(anno_lbl2_ctype[i]))+ ',' + ctype + ',' + str(pred_cell_count_ctype[i])  + ',' + str(pred_patch_count_ctype[i])   ) ;
                file.write(anno_arr_ctype[i,1]+'_'+anno_arr_ctype[i,2] + ','+str(int(anno_lbl2_ctype[i]))+ ',' + ctype  + ',' + str(pred_patch_count_ctype[i])   ) ;
                for j in range(anno_arr_lbl_full_ctype.shape[-1]):
                    file.write(',' + anno_arr_lbl_full_ctype[i,j]) ;            
                file.write('\n') ;

def process_results_violin_use_anno_csv(in_dir, out_dir, csv_path, model_prefix, dataset_name, threshold, plot_type=1):

    anno_arr = np.loadtxt(csv_path, delimiter=',', dtype=str)
    result_files_prefix = os.path.join(in_dir, model_prefix );
    out_files_prefix = os.path.join(in_dir, model_prefix + '_'+dataset_name);
    #lbl = np.load(result_files_prefix + '_individual_labels.npy', allow_pickle=True);
    anno_lbl = anno_arr[:, 2:]
    print('anno_lbl',anno_lbl.shape)
    anno_lbl[anno_lbl=='']='0'
    anno_lbl[anno_lbl==' ']='0'
    anno_lbl = anno_lbl.astype(int)
    #pred = np.load(result_files_prefix + '_pred_new.npy');
    if(os.path.isfile(result_files_prefix + '_pred_new.npy')):
        pred = np.load(result_files_prefix + '_pred_new.npy', allow_pickle=True);
    elif(os.path.isfile(result_files_prefix + '_pred_prob.npy')):
        pred = np.load(result_files_prefix + '_pred_prob.npy', allow_pickle=True);    
    filenames = pickle.load(open(result_files_prefix + '_filename.pkl', 'rb'));    
    #print('pred.shape = ', pred.shape)
    pred= pred.squeeze();
    print('pred.shape = ', pred.shape)
    #pred = pred[:,:,1] ;
    if(len(pred.shape) > 2 and pred.shape[2]>1):
        pred = pred[:,:,1];
    elif(len(pred.shape) > 2 and pred.shape[2]==1):
        pred = pred[:,:,0];
    elif(len(pred.shape) == 2):
        pred = pred;
    lbl = np.zeros((pred.shape[0], anno_lbl.shape[-1]))
    for i in range (len(filenames)):
        f = filenames[i]
        #anno_row = anno_arr[np.where(anno_arr[:,1]==f)]
        lbl[i] = anno_lbl[np.where(anno_arr[:,1]==f)]
    ctype = pickle.load(open(result_files_prefix + '_cancer_type.pkl', 'rb'));
    ctype = np.array(ctype);
    if(not (exclude_ctype is None)):
        ctype = pickle.load(open(result_files_prefix + '_cancer_type.pkl', 'rb'));
        ctype = np.array(ctype);
        pred = pred[np.where(ctype!=exclude_ctype)]
        lbl = lbl[np.where(ctype!=exclude_ctype)]
        ctype = ctype[np.where(ctype!=exclude_ctype)]
    print('include_ctype=',include_ctype)
    if(not (include_ctype is None)):
        ctype = pickle.load(open(result_files_prefix + '_cancer_type.pkl', 'rb'));
        ctype = np.array(ctype);
        pred = pred[np.where(ctype==include_ctype)]
        lbl = lbl[np.where(ctype==include_ctype)]
        ctype = ctype[np.where(ctype==include_ctype)]
        print(np.where(ctype==include_ctype)[0])
        filenames = np.array(filenames)[np.where(ctype==include_ctype)]
        print(filenames)

        

    # label values are: 1, 2, 3, 4,; empty,  ignore value of 4 and empty
    lbl[np.where(lbl==4)] = 0   ;
    # to get the average score label need to get the count of scores available for each super patch
    b = lbl>0;
    n = b.sum(axis = 1);
    n[np.where(n ==0)] = -1; # set to -1 the count = 0 to avoid division by zero
    # get the average score label by summing each patch scores and divide by count then round
    lbl2 = np.divide(lbl.sum(axis = 1), n);
    lbl2 = np.round(lbl2);

    print('lbl=1', len(np.where(lbl2 == 1)[0])) # 23
    print('lbl=2', len(np.where(lbl2 == 2)[0])) # 29
    print('lbl=3', len(np.where(lbl2 == 3)[0])) # 11

    # get the sub patches that are predicted positive according to threshold
    # the pred is super patch -> sub patch -> logit neg, logit pos
    pred_b = pred > threshold ;

    # get the number of subpatches predicted positive in each superpatch
    pred_n = pred_b.sum(axis = 1)
    print('np.unique(pred_n)',np.unique(pred_n))
    # get the number of subpatches predicted positive in each superpatch in each score label category 1,2,3
    pred_n1 = pred_n[np.where(lbl2 == 1)]
    pred_n2 = pred_n[np.where(lbl2 == 2)]
    pred_n3 = pred_n[np.where(lbl2 == 3)]

    ctype_name = ''
    if(not (include_ctype is None)):
        ctype_name = '_'+include_ctype
    with open(os.path.join(out_dir, model_prefix +ctype_name+ '_lbl.txt'), 'w') as file:
        for i in range(len(lbl)):
            file.write(filenames[i] + ',' + str(pred_n[i]) + ','+str(int(lbl2[i]))+ ',' + str(ctype[i])+'\n');

    #print(pred_n1) ;
    #print(np.where(lbl2 == 1)) ;
    #print(lbl[np.where(lbl2 == 1)]) ;

    if(plot_type == 0 or plot_type == 1 or plot_type == 2):
        if(not(0 in pred_n1)):
            pred_n1 = np.concatenate((pred_n1, [0]))

        if(not(64 in pred_n1)):
            pred_n1 = np.concatenate((pred_n1, [64]))

        if(not(0 in pred_n2)):
            pred_n2 = np.concatenate((pred_n2, [0]))

        if(not(64 in pred_n2)):
            pred_n2 = np.concatenate((pred_n2, [64]))

        if(not(0 in pred_n3)):
            pred_n3 = np.concatenate((pred_n3, [0]))

        if(not(64 in pred_n3)):
            pred_n3 = np.concatenate((pred_n3, [64]))

    fig,ax = plt.subplots(1)
    sns.set(style="whitegrid")
    #data = {'pred_n':pred_n}
    #sns.violinplot(y=pred_n3, bw=1) # multiplies bw by the std to control smoothness
    #sns.violinplot(y=pred_n3, bw=1, cut=0) # cut =0 means do not extend beyond data range default is 2
    #sns.violinplot(y=pred_n3, bw=1, cut=0, scale='count') # scale reflects the relative shapes of the different violins 'width:same width, area:same area, count:width relative to count in category'
    #sns.violinplot(y=pred_n3, bw=1, cut=0, scale='width')
    #sns.violinplot(y=pred_n3, bw=1, cut=0, width=0.5) # the width of the violin default is 0.8
    if(plot_type == 0):
        ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], cut=0, width=0.7, scale='width', ax=ax)
    elif(plot_type == 1):
        ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], bw=1, cut=0, width=0.5, scale='width', ax=ax) # original (1)
    elif(plot_type == 2 or plot_type == 3):
        ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], cut=0, ax=ax)
    ax.set(xticklabels=['low', 'medium', 'high'])
    #plt.show();
    fig.savefig(os.path.join(out_dir, model_prefix +'_' +dataset_name+'_violin'+'_type'+str(plot_type)+'_review.png'));
    return;


def process_results_violin(in_dir, out_dir, model_prefix, dataset_name, threshold, plot_type=1, exclude_ctype=None, include_ctype=None):

    result_files_prefix = os.path.join(in_dir, model_prefix );
    out_files_prefix = os.path.join(in_dir, model_prefix + '_'+dataset_name);
    lbl = np.load(result_files_prefix + '_individual_labels.npy');
    #pred = np.load(result_files_prefix + '_pred_new.npy');
    if(os.path.isfile(result_files_prefix + '_pred_new.npy')):
        pred = np.load(result_files_prefix + '_pred_new.npy', allow_pickle=True);
    elif(os.path.isfile(result_files_prefix + '_pred_prob.npy')):
        pred = np.load(result_files_prefix + '_pred_prob.npy', allow_pickle=True);    
    filenames = pickle.load(open(result_files_prefix + '_filename.pkl', 'rb'));
    #print('pred.shape = ', pred.shape)
    pred= pred.squeeze();
    print('pred.shape = ', pred.shape)
    #pred = pred[:,:,1] ;
    if(len(pred.shape) > 2 and pred.shape[2]>1):
        pred = pred[:,:,1];
    elif(len(pred.shape) > 2 and pred.shape[2]==1):
        pred = pred[:,:,0];
    elif(len(pred.shape) == 2):
        pred = pred;
    ctype = pickle.load(open(result_files_prefix + '_cancer_type.pkl', 'rb'));
    ctype = np.array(ctype);
    if(not (exclude_ctype is None)):
        ctype = pickle.load(open(result_files_prefix + '_cancer_type.pkl', 'rb'));
        ctype = np.array(ctype);
        pred = pred[np.where(ctype!=exclude_ctype)]
        lbl = lbl[np.where(ctype!=exclude_ctype)]
        ctype = ctype[np.where(ctype!=exclude_ctype)]
    print('include_ctype=',include_ctype)
    if(not (include_ctype is None)):
        ctype = pickle.load(open(result_files_prefix + '_cancer_type.pkl', 'rb'));
        ctype = np.array(ctype);
        pred = pred[np.where(ctype==include_ctype)]
        lbl = lbl[np.where(ctype==include_ctype)]
        ctype = ctype[np.where(ctype==include_ctype)]
        print(np.where(ctype==include_ctype)[0])
        filenames = np.array(filenames)[np.where(ctype==include_ctype)]
        print(filenames)

        

    # label values are: 1, 2, 3, 4,; empty,  ignore value of 4 and empty
    lbl[np.where(lbl==4)] = 0   ;
    # to get the average score label need to get the count of scores available for each super patch
    b = lbl>0;
    n = b.sum(axis = 1);
    n[np.where(n ==0)] = -1; # set to -1 the count = 0 to avoid division by zero
    # get the average score label by summing each patch scores and divide by count then round
    lbl2 = np.divide(lbl.sum(axis = 1), n);
    lbl2 = np.round(lbl2);

    print('lbl=1', len(np.where(lbl2 == 1)[0])) # 23
    print('lbl=2', len(np.where(lbl2 == 2)[0])) # 29
    print('lbl=3', len(np.where(lbl2 == 3)[0])) # 11

    # get the sub patches that are predicted positive according to threshold
    # the pred is super patch -> sub patch -> logit neg, logit pos
    pred_b = pred > threshold ;

    # get the number of subpatches predicted positive in each superpatch
    pred_n = pred_b.sum(axis = 1)
    print('np.unique(pred_n)',np.unique(pred_n))
    # get the number of subpatches predicted positive in each superpatch in each score label category 1,2,3
    pred_n1 = pred_n[np.where(lbl2 == 1)]
    pred_n2 = pred_n[np.where(lbl2 == 2)]
    pred_n3 = pred_n[np.where(lbl2 == 3)]

    ctype_name = ''
    if(not (include_ctype is None)):
        ctype_name = '_'+include_ctype
    with open(os.path.join(out_dir, model_prefix +ctype_name+ '_lbl.txt'), 'w') as file:
        for i in range(len(lbl)):
            file.write(filenames[i] + ',' + str(pred_n[i]) + ','+str(int(lbl2[i])) + ',' + str(ctype[i])+'\n');

    #print(pred_n1) ;
    #print(np.where(lbl2 == 1)) ;
    #print(lbl[np.where(lbl2 == 1)]) ;

    if(plot_type == 0 or plot_type == 1 or plot_type == 2):
        if(not(0 in pred_n1)):
            pred_n1 = np.concatenate((pred_n1, [0]))

        if(not(64 in pred_n1)):
            pred_n1 = np.concatenate((pred_n1, [64]))

        if(not(0 in pred_n2)):
            pred_n2 = np.concatenate((pred_n2, [0]))

        if(not(64 in pred_n2)):
            pred_n2 = np.concatenate((pred_n2, [64]))

        if(not(0 in pred_n3)):
            pred_n3 = np.concatenate((pred_n3, [0]))

        if(not(64 in pred_n3)):
            pred_n3 = np.concatenate((pred_n3, [64]))

    fig,ax = plt.subplots(1)
    sns.set(style="whitegrid")
    #data = {'pred_n':pred_n}
    #sns.violinplot(y=pred_n3, bw=1) # multiplies bw by the std to control smoothness
    #sns.violinplot(y=pred_n3, bw=1, cut=0) # cut =0 means do not extend beyond data range default is 2
    #sns.violinplot(y=pred_n3, bw=1, cut=0, scale='count') # scale reflects the relative shapes of the different violins 'width:same width, area:same area, count:width relative to count in category'
    #sns.violinplot(y=pred_n3, bw=1, cut=0, scale='width')
    #sns.violinplot(y=pred_n3, bw=1, cut=0, width=0.5) # the width of the violin default is 0.8
    if(plot_type == 0):
        ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], cut=0, width=0.7, scale='width', ax=ax)
    elif(plot_type == 1):
        ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], bw=1, cut=0, width=0.5, scale='width', ax=ax) # original (1)
    elif(plot_type == 2 or plot_type == 3):
        ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], cut=0, ax=ax)
    ax.set(xticklabels=['low', 'medium', 'high'])
    #plt.show();
    fig.savefig(os.path.join(out_dir, model_prefix +'_' +dataset_name+'_violin'+'_type'+str(plot_type)+'.png'));
    return;

def save_n_pred_pos(in_dir, model_prefix, threshold):

    result_files_prefix = os.path.join(in_dir, model_prefix );
    #pred = np.load(result_files_prefix + '_pred_new.npy');
    if(os.path.isfile(result_files_prefix + '_pred_new.npy')):
        pred = np.load(result_files_prefix + '_pred_new.npy', allow_pickle=True);
    elif(os.path.isfile(result_files_prefix + '_pred_prob.npy')):
        pred = np.load(result_files_prefix + '_pred_prob.npy', allow_pickle=True);    

    # get the sub patches that are predicted positive according to threshold
    # the pred is super patch -> sub patch -> logit neg, logit pos
    pred= pred.squeeze();
    #pred = pred[:,:,1] ;
    if(len(pred.shape) > 2 and pred.shape[2]>1):
        pred = pred[:,:,1];
    elif(len(pred.shape) > 2 and pred.shape[2]==1):
        pred = pred[:,:,0];
    elif(len(pred.shape) == 2):
        pred = pred;
    pred_b = pred > threshold ;

    # get the number of subpatches predicted positive in each superpatch
    pred_n = pred_b.sum(axis = 1)

    pred_n.dump(result_files_prefix + '_pred_n.npy');
  
    return;

def process_results_violin_old_model(in_dir, out_dir, model_prefix, dataset_name, plot_type=1, exclude_ctype=None):

    result_files_prefix = os.path.join(in_dir, model_prefix);
    out_files_prefix = os.path.join(in_dir, model_prefix);
    lbl = np.load(result_files_prefix + '_individual_labels.npy', allow_pickle=True);
    pred_n_str = np.load(result_files_prefix + '_pred_old.npy', allow_pickle=True);
    if(not (exclude_ctype is None)):
        ctype = pickle.load(open(result_files_prefix + '_cancer_type.pkl', 'rb'));
        ctype = np.array(ctype);
        pred_n_str = pred_n_str[np.where(ctype!=exclude_ctype)]
        lbl = lbl[np.where(ctype!=exclude_ctype)]

    # label values are: 1, 2, 3, 4,; empty,  ignore value of 4 and empty
    lbl[np.where(lbl==4)] = 0   ;
    # to get the average score label need to get the count of scores available for each super patch
    b = lbl>0;
    n = b.sum(axis = 1);
    n[np.where(n ==0)] = -1; # set to -1 the count = 0 to avoid division by zero
    # get the average score label by summing each patch scores and divide by count then round
    lbl2 = np.divide(lbl.sum(axis = 1), n);
    lbl2 = np.round(lbl2);

    ## get the sub patches that are predicted positive according to threshold
    ## the pred is super patch -> sub patch -> logit neg, logit pos
    #pred= pred.squeeze();
    #pred = pred[:,:,1] ;
    #pred_b = pred > threshold ;

    # get the number of subpatches predicted positive in each superpatch
    #pred_n = pred_b.sum(axis = 1)
    pred_n = pred_n_str.astype(np.int)
    # get the number of subpatches predicted positive in each superpatch in each score label category 1,2,3
    pred_n1 = pred_n[np.where(lbl2 == 1)]
    pred_n2 = pred_n[np.where(lbl2 == 2)]
    pred_n3 = pred_n[np.where(lbl2 == 3)]

    if(plot_type == 0 or plot_type == 1 or plot_type == 2):
        if(not(0 in pred_n1)):
            pred_n1 = np.concatenate((pred_n1, [0]))

        if(not(64 in pred_n1)):
            pred_n1 = np.concatenate((pred_n1, [64]))

        if(not(0 in pred_n2)):
            pred_n2 = np.concatenate((pred_n2, [0]))

        if(not(64 in pred_n2)):
            pred_n2 = np.concatenate((pred_n2, [64]))

        if(not(0 in pred_n3)):
            pred_n3 = np.concatenate((pred_n3, [0]))

        if(not(64 in pred_n3)):
            pred_n3 = np.concatenate((pred_n3, [64]))

    fig,ax = plt.subplots(1)
    sns.set(style="whitegrid")
    #ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], bw=0.5, cut=0, width=0.5, scale='width', ax=ax)
    #ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], cut=0, width=0.5, scale='width', ax=ax)
    if(plot_type == 0):
        ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], cut=0, width=0.7, scale='width', ax=ax)
    elif(plot_type == 1):
        ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], bw=1, cut=0, width=0.5, scale='width', ax=ax) # original (1)
    elif(plot_type == 2 or plot_type == 3):
        ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], cut=0, ax=ax)
    ax.set(xticklabels=['low', 'medium', 'high'])
    #plt.show();
    fig.savefig('baseline.png');
    return;

def process_results_violin_old_model_w_thresh26(csv_path, plot_type=1, exclude_ctype=None):

    cancer_type_list = [];
    filename_list = [];
    individual_labels_list = []
    avg_label_list = []
    pred_old_list = []

    # read csv file
    with open(csv_path, 'r') as label_file:
        line = label_file.readline(); # skip title line
        line = label_file.readline();
        while(line):
            c, s, p, i1, i2, i3, i4, i5, i6, pred_old, pred_thresh23_nec, pred_thresh23, pred_thresh26_nec, pred_thresh26= line.split(',');
            if (i1.strip()==""):
                i1 = 0;
            if (i2.strip()==""):
                i2 = 0;
            if (i3.strip()==""):
                i3 = 0;
            if (i4.strip()==""):
                i4 = 0;
            if (i5.strip()==""):
                i5 = 0;
            if (i6.strip()==""):
                i6 = 0;
            cancer_type_list.append(c);
            filename_list.append(s+'_'+p+'.png');
            individual_labels_list.append([int(i1), int(i2), int(i3), int(i4), int(i5), int(i6)]);
            avg_label_list.append(np.mean(np.array([float(i1), float(i2), float(i3), float(i4), float(i5), float(i6)])));
            #pred_old_list.append(pred_old);
            pred_old_list.append(pred_thresh26);
            line = label_file.readline();

    lbl = np.array(individual_labels_list);
    pred_n_str = np.array(pred_old_list);
    if(not (exclude_ctype is None)):
        ctype = np.array(cancer_type_list);
        pred_n_str = pred_n_str[np.where(ctype!=exclude_ctype)]
        lbl = lbl[np.where(ctype!=exclude_ctype)]

    # label values are: 1, 2, 3, 4,; empty,  ignore value of 4 and empty
    lbl[np.where(lbl==4)] = 0   ;
    # to get the average score label need to get the count of scores available for each super patch
    b = lbl>0;
    n = b.sum(axis = 1);
    n[np.where(n ==0)] = -1; # set to -1 the count = 0 to avoid division by zero
    # get the average score label by summing each patch scores and divide by count then round
    lbl2 = np.divide(lbl.sum(axis = 1), n);
    lbl2 = np.round(lbl2);

    ## get the sub patches that are predicted positive according to threshold
    ## the pred is super patch -> sub patch -> logit neg, logit pos
    #pred= pred.squeeze();
    #pred = pred[:,:,1] ;
    #pred_b = pred > threshold ;

    # get the number of subpatches predicted positive in each superpatch
    #pred_n = pred_b.sum(axis = 1)
    pred_n = pred_n_str.astype(np.int)
    # get the number of subpatches predicted positive in each superpatch in each score label category 1,2,3
    pred_n1 = pred_n[np.where(lbl2 == 1)]
    pred_n2 = pred_n[np.where(lbl2 == 2)]
    pred_n3 = pred_n[np.where(lbl2 == 3)]

    if(plot_type == 0 or plot_type == 1 or plot_type == 2):
        if(not(0 in pred_n1)):
            pred_n1 = np.concatenate((pred_n1, [0]))

        if(not(64 in pred_n1)):
            pred_n1 = np.concatenate((pred_n1, [64]))

        if(not(0 in pred_n2)):
            pred_n2 = np.concatenate((pred_n2, [0]))

        if(not(64 in pred_n2)):
            pred_n2 = np.concatenate((pred_n2, [64]))

        if(not(0 in pred_n3)):
            pred_n3 = np.concatenate((pred_n3, [0]))

        if(not(64 in pred_n3)):
            pred_n3 = np.concatenate((pred_n3, [64]))

    fig,ax = plt.subplots(1)
    sns.set(style="whitegrid")
    #ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], bw=0.5, cut=0, width=0.5, scale='width')
    #ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], cut=0, width=0.5, scale='width')
    if(plot_type == 0):
        ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], cut=0, width=0.7, scale='width', ax=ax)
    elif(plot_type == 1):
        ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], bw=1, cut=0, width=0.5, scale='width', ax=ax) # original (1)
    elif(plot_type == 2 or plot_type == 3):
        ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], cut=0, ax=ax)
    ax.set(xticklabels=['low', 'medium', 'high'])
    #plt.show();
    fig.savefig('baseline_th26.png');
    return;

def process_results_violin_han(label_filepath, pred_filepath, out_dir, model_prefix, dataset_name, plot_type=3, ctype='brca'):

    cancer_type_list = [];
    filename_list = [];
    individual_labels_list = []

    with open(os.path.join(label_filepath), 'r') as label_file:
        line = label_file.readline();
        line = label_file.readline();
        while(line):
            c, s, p, i1, i2, i3, i4, i5, i6, pred_old= line.split(',');
            print(c,s,p)
            if(not (ctype is None) and not (c.strip() == ctype)):
                line = label_file.readline();
                continue;
            if (i1.strip()==""):
                i1 = 0;
            if (i2.strip()==""):
                i2 = 0;
            if (i3.strip()==""):
                i3 = 0;
            if (i4.strip()==""):
                i4 = 0;
            if (i5.strip()==""):
                i5 = 0;
            if (i6.strip()==""):
                i6 = 0;
            cancer_type_list.append(c);
            filename_list.append(s+'_'+p+'.png');
            individual_labels_list.append([int(i1), int(i2), int(i3), int(i4), int(i5), int(i6)]);
            line = label_file.readline();


    pred_filename_list = [];
    pred_n_list = [];
    pred_individual_labels = [];
    with open(os.path.join(pred_filepath), 'r') as file:
        line = file.readline();
        while(line):            
            s, pred = line.split(',');
            print(s, pred)
            pred_filename_list.append(s);
            pred_n_list.append(int(pred));
            line = file.readline();

    pred_n = np.array(pred_n_list);
    for i in range(len(pred_filename_list)):
        patch_filename = pred_filename_list[i].strip();
        print(patch_filename )
        for j in range(len(filename_list)):
            if(filename_list[j].strip() == patch_filename):
                print('found')
                pred_individual_labels.append(individual_labels_list[j]);
                break;

    lbl = np.array(pred_individual_labels);



    # label values are: 1, 2, 3, 4,; empty,  ignore value of 4 and empty
    lbl[np.where(lbl==4)] = 0   ;
    # to get the average score label need to get the count of scores available for each super patch
    b = lbl>0;
    n = b.sum(axis = 1);
    n[np.where(n ==0)] = -1; # set to -1 the count = 0 to avoid division by zero
    # get the average score label by summing each patch scores and divide by count then round
    lbl2 = np.divide(lbl.sum(axis = 1), n);
    lbl2 = np.round(lbl2);

    ## get the sub patches that are predicted positive according to threshold
    ## the pred is super patch -> sub patch -> logit neg, logit pos
    #pred= pred.squeeze();
    #pred = pred[:,:,1] ;
    #pred_b = pred > threshold ;

    # get the number of subpatches predicted positive in each superpatch
    #pred_n = pred_b.sum(axis = 1)
    #pred_n = pred_n_str.astype(np.int)
    # get the number of subpatches predicted positive in each superpatch in each score label category 1,2,3
    pred_n1 = pred_n[np.where(lbl2 == 1)]
    pred_n2 = pred_n[np.where(lbl2 == 2)]
    pred_n3 = pred_n[np.where(lbl2 == 3)]
    print('lbl=1', len(np.where(lbl2 == 1)[0])) # 23
    print('lbl=2', len(np.where(lbl2 == 2)[0])) # 29
    print('lbl=3', len(np.where(lbl2 == 3)[0])) # 11

    if(plot_type == 0 or plot_type == 1 or plot_type == 2):
        if(not(0 in pred_n1)):
            pred_n1 = np.concatenate((pred_n1, [0]))

        if(not(64 in pred_n1)):
            pred_n1 = np.concatenate((pred_n1, [64]))

        if(not(0 in pred_n2)):
            pred_n2 = np.concatenate((pred_n2, [0]))

        if(not(64 in pred_n2)):
            pred_n2 = np.concatenate((pred_n2, [64]))

        if(not(0 in pred_n3)):
            pred_n3 = np.concatenate((pred_n3, [0]))

        if(not(64 in pred_n3)):
            pred_n3 = np.concatenate((pred_n3, [64]))

    fig,ax = plt.subplots(1)
    sns.set(style="whitegrid")
    #ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], bw=0.5, cut=0, width=0.5, scale='width', ax=ax)
    #ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], cut=0, width=0.5, scale='width', ax=ax)
    if(plot_type == 0):
        ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], cut=0, width=0.7, scale='width', ax=ax)
    elif(plot_type == 1):
        ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], bw=1, cut=0, width=0.5, scale='width', ax=ax) # original (1)
    elif(plot_type == 2 or plot_type == 3):
        ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], cut=0, ax=ax)
    ax.set(xticklabels=['low', 'medium', 'high'])
    #plt.show();
    fig.savefig(os.path.join(out_dir, model_prefix +'_' +dataset_name+'_violin'+'_type'+str(plot_type)+'.png'));
    return;


if __name__ == "__main__":

    #model_prefix = "tcga_incv4_mix_new3"; 
    #threshold = 0.41;
    #in_dir = "/home/shahira/TIL_classification/superpatch_merge";
    #out_dir = "/home/shahira/TIL_classification/eval_superpatch_merge/incep_poly_th0.4"    

    #model_prefix = "tcga_vgg16_mix_new3"; 
    #threshold = 0.4;
    #in_dir  = "/home/shahira/TIL_classification/eval_superpatch_merge"    
    #out_dir = "/home/shahira/TIL_classification/eval_superpatch_merge/vgg_poly_th0.4"    

    model_prefix = ""
    threshold = 0.56
    in_dir  = "/home/shahira/TIL_classification/eval_superpatch_merge/resnet34_e12"    
    out_dir = "/home/shahira/TIL_classification/eval_superpatch_merge/resnet_poly_th0.56"    


    dataset_name = "superpatches_merged"
    csv_path = '/home/shahira/TIL_classification/superpatch_merge/super-patches-label_m.csv'
    title_line = 'filename,label,ctype,patch_count,Anne1,Anne2,Raj1,Raj2,Rebecca1,Rebecca2'
    process_results_violin_use_anno_csv_outraw_merged(in_dir, out_dir, csv_path, model_prefix, dataset_name, threshold, title_line);

    #out_dir = "/home/shahira/TIL_classification/eval_superpatch_merge/le_poly";
    #process_le_results_violin_use_anno_csv_outraw_merged(out_dir, csv_path, dataset_name, title_line)

    #############################################################################################################
    #model_prefix = "tcga_incv4_mix_new3"; 
    #threshold = 0.41;
    #in_dir = "/home/shahira/TIL_classification/superpatches_anno/superpatches_eval";
    #out_dir = "/home/shahira/TIL_classification/superpatches_anno/superpatches_eval/poly_th0.4"    

    #model_prefix = "tcga_vgg16_mix_new3"; 
    #threshold = 0.4;
    #in_dir = "/home/shahira/TIL_classification/superpatches_anno/superpatches_eval";
    #out_dir = "/home/shahira/TIL_classification/superpatches_anno/superpatches_eval/vgg_poly_th0.4"    

    #model_prefix = ""; 
    #threshold = 0.56;
    #in_dir = "/home/shahira/TIL_classification/superpatches_anno/superpatches_eval/resnet34_e12";
    #out_dir = "/home/shahira/TIL_classification/superpatches_anno/superpatches_eval/resnet34_e12/poly_th0.56"    

    #dataset_name = "superpatches_review"
    #csv_path = '/home/shahira/TIL_classification/superpatches_anno/anno_reviewed_individual.csv'
    #title_line = 'filename,label,ctype,patch_count,John,Anne,Rebecca'
    #process_results_violin_use_anno_csv_outraw_reviewed(in_dir, out_dir, csv_path, model_prefix, dataset_name, threshold, title_line);

