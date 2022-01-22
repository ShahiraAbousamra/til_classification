import numpy as np;
import pickle;
import os;
import sys;
from sklearn import metrics 
#import matplotlib.pyplot as plt
from shutil import copyfile;
import glob;

def copy_fn(in_dir, model_prefix, img_root_dir, img_root_dir_rect, img_out_dir, threshold, types):
#def copy_fn(in_dir, model_prefix, img_root_dir, img_out_dir, threshold):
    print('in copy_fn')
    sys.stdout.flush();
    #types = ['luad', 'brca', 'acc', 'kirc', 'lihc', 'ov', 'hnsc'];

    result_files_prefix = os.path.join(in_dir, model_prefix);
    lbl = np.load(result_files_prefix + '_individual_labels.npy', allow_pickle=True);
    if(os.path.isfile(result_files_prefix + '_pred_new.npy')):
        pred = np.load(result_files_prefix + '_pred_new.npy', allow_pickle=True);
    elif(os.path.isfile(result_files_prefix + '_pred_prob.npy')):
        pred = np.load(result_files_prefix + '_pred_prob.npy', allow_pickle=True);    
    pred = pred.squeeze();    
    print('pred.shape = ', pred.shape);
    sys.stdout.flush();
    if(len(pred.shape) > 1 and pred.shape[1]>1):
        pred_t1 = pred[:,1];
    elif(len(pred.shape) > 1 and pred.shape[1]==1):
        pred_t1 = pred[:,0];
    elif(len(pred.shape) == 1):
        pred_t1 = pred;

    filenames = pickle.load(open(result_files_prefix + '_filename.pkl', "rb"));
    filenames = np.array(filenames)

    pred_lbl_pos = pred_t1[np.where(lbl > 0)]
    filenames_lbl_pos = filenames[np.where(lbl > 0)]
    #print(pred_lbl_pos.shape)
    #print(filenames_lbl_pos.shape)
    pred_fn_thresh = pred_lbl_pos[np.where(pred_lbl_pos<threshold)]
    filenames_fn_thresh = filenames_lbl_pos[np.where(pred_lbl_pos<threshold)]
    #print(pred_fn_thresh.shape)
    #print(filenames_fn_thresh.shape)

    prefix = ''
    for file in filenames_fn_thresh:
        print(file)
        prefix = ''
        for type in types:
            if(type in file):
                prefix = type + '_'
                break
        file = file.replace(img_root_dir, img_root_dir_rect)
        copyfile(file, os.path.join(img_out_dir, prefix + os.path.basename(file)), follow_symlinks=True)



def copy_fp(in_dir, model_prefix, img_root_dir, img_root_dir_rect, img_out_dir, threshold, types):
#def copy_fn(in_dir, model_prefix, img_root_dir, img_out_dir, threshold):
    print('in copy_fp')
    sys.stdout.flush();
    #types = ['luad', 'brca', 'acc', 'kirc', 'lihc', 'ov', 'hnsc'];

    result_files_prefix = os.path.join(in_dir, model_prefix);
    lbl = np.load(result_files_prefix + '_individual_labels.npy', allow_pickle=True);
    if(os.path.isfile(result_files_prefix + '_pred_new.npy')):
        pred = np.load(result_files_prefix + '_pred_new.npy', allow_pickle=True);
    elif(os.path.isfile(result_files_prefix + '_pred_prob.npy')):
        pred = np.load(result_files_prefix + '_pred_prob.npy', allow_pickle=True);    
    pred = pred.squeeze();    
    print('pred.shape = ', pred.shape);
    sys.stdout.flush();
    if(len(pred.shape) > 1 and pred.shape[1]>1):
        pred_t1 = pred[:,1];
    elif(len(pred.shape) > 1 and pred.shape[1]==1):
        pred_t1 = pred[:,0];
    elif(len(pred.shape) == 1):
        pred_t1 = pred;

    filenames = pickle.load(open(result_files_prefix + '_filename.pkl', "rb"));
    filenames = np.array(filenames)

    pred_lbl_neg = pred_t1[np.where(lbl < 1)]
    filenames_lbl_neg = filenames[np.where(lbl < 1)]
    #print(pred_lbl_neg.shape)
    #print(filenames_lbl_neg.shape)
    pred_fp_thresh = pred_lbl_neg[np.where(pred_lbl_neg>threshold)]
    filenames_fp_thresh = filenames_lbl_neg[np.where(pred_lbl_neg>threshold)]
    #print(pred_fp_thresh.shape)
    #print(filenames_fp_thresh.shape)

    prefix = ''
    for file in filenames_fp_thresh:
        print(file)
        prefix = ''
        for type in types:
            if(type in file):
                prefix = type + '_'
                break
        file = file.replace(img_root_dir, img_root_dir_rect)
        copyfile(file, os.path.join(img_out_dir, prefix + os.path.basename(file)), follow_symlinks=True)




def process_results_separate_types(in_dir, out_dir, model_prefix, threshold, types):
    #types = ['coad', 'brca', 'read', 'luad', 'uvm', 'lusc', 'ucec', 'stad', 'blca', 'paad', 'prad', 'cesc', 'skcm', 'brca_additional+', 'luad_additional+'];
    #types = ['coad', 'read', 'uvm', 'lusc', 'ucec', 'stad', 'blca', 'paad', 'prad', 'cesc', 'skcm', 'brca_additional+', 'luad_additional+'];
    #types = ['val_luad_strat', 'val_brca', 'val_acc', 'val_kirc', 'val_lihc', 'val_hnsc', 'val_ov'];
    #types = ['val_luad_strat', 'val_brca', 'val_acc', 'val_kirc', 'val_lihc', 'val_ov'];
    #types = ['val_hnsc'];
    out_filename = os.path.join(in_dir, model_prefix + '_ctypes_stats_th-'+ str(threshold)+'.csv');
    out_file = open(out_filename, 'w+');
    print(out_filename);
    out_file.write(model_prefix+'\n\n');
    TP_total = 0;
    FP_total = 0;
    TN_total = 0;
    FN_total = 0;
    T_total = 0;
    total = 0;
    for ctype in types:
        print(ctype);
        result_files_prefix = os.path.join(in_dir, ctype, model_prefix);
        lbl = np.load(result_files_prefix + '_individual_labels.npy', allow_pickle=True);        
        #pred = np.load(result_files_prefix + '_pred_new.npy', allow_pickle=True);
        if(os.path.isfile(result_files_prefix + '_pred_new.npy')):
            pred = np.load(result_files_prefix + '_pred_new.npy', allow_pickle=True);
        elif(os.path.isfile(result_files_prefix + '_pred_prob.npy')):
            pred = np.load(result_files_prefix + '_pred_prob.npy', allow_pickle=True);    
        filenames = pickle.load(open(result_files_prefix + '_filename.pkl', "rb"));
        filenames = np.array(filenames);
        pred = pred.squeeze();
        #pred_t1 = pred[:,1];
        if(len(pred.shape) > 1 and pred.shape[1]>1):
            pred_t1 = pred[:,1];
        elif(len(pred.shape) > 1 and pred.shape[1]==1):
            pred_t1 = pred[:,0];
        elif(len(pred.shape) == 1):
            pred_t1 = pred;
        count = lbl.shape[0];
        P = lbl[np.where(pred_t1 > threshold)].shape[0]
        TP = lbl[np.where(pred_t1 > threshold)].sum();
        FP = P - TP;
        N = count - P;
        FN = lbl[np.where(pred_t1 <= threshold)].sum();
        TN = N - FN;
        lbl_p = lbl[np.where(pred_t1 > threshold)];
        #print(filenames[np.where(lbl_p == 0)]);
        print("TP = ", TP);
        print("FP = ", FP);
        print("TN = ", TN);
        print("FN = ", FN);
        prec = TP/float(TP + FP);
        recall = TP/float(TP + FN);
        print("prec = ", prec);
        print("recall = ", recall);
        f1 = 2/(1/prec + 1/recall);
        accuracy = (TP+TN)/float(count);
        out_file.write(ctype + ',' + "Pred. Positive" + "," + "Pred. Negative" + "," + "Prec."+ "," + "Recall"+ "," + "F1" + "," + "accuracy" +'\r\n');
        out_file.write("Label Positive" + "," + str(TP) + "," + str(FN) + "," + "{:.2f}".format(prec) + "," + "{:.2f}".format(recall) + "," + "{:.2f}".format(f1) + "," + "{:.2f}".format(accuracy) +'\r\n');
        out_file.write("Label Negative" + "," + str(FP) + "," + str(TN) +'\r\n');
        TP_total += TP;
        FP_total += FP;
        TN_total += TN;
        FN_total += FN;
        T_total += TP + TN;
        total += count;

    prec = TP_total/float(TP_total + FP_total);
    recall = TP_total/float(TP_total + FN_total);
    f1 = 2/(1/prec + 1/recall);
    accuracy = (T_total)/float(total);

    out_file.write('\n\n');
    out_file.write('All' + ',' + "Pred. Positive" + "," + "Pred. Negative" + "," + "Prec."+ "," + "Recall"+ "," + "F1" + "," + "accuracy" +'\r\n');
    out_file.write("Label Positive" + "," + str(TP_total) + "," + str(FN_total) + "," + "{:.2f}".format(prec) + "," + "{:.2f}".format(recall) + "," + "{:.2f}".format(f1) + "," + "{:.2f}".format(accuracy) +'\r\n');
    out_file.write("Label Negative" + "," + str(FP_total) + "," + str(TN_total) +'\r\n');

    out_file.close();

def process_results_separate_types_sm(in_dir, out_dir, model_prefix, threshold, types):
    #types = ['coad', 'brca', 'read', 'luad', 'uvm', 'lusc', 'ucec', 'stad', 'blca', 'paad', 'prad', 'cesc', 'skcm'];
    #types = ['ucec'];
    out_filename = os.path.join(in_dir, model_prefix + '_ctypes_stats_th-'+ str(threshold)+'_sm.csv');
    out_file = open(out_filename, 'w+');
    print(out_filename);
    out_file.write(model_prefix+'\n\n');
    TP_total = 0;
    FP_total = 0;
    TN_total = 0;
    FN_total = 0;
    T_total = 0;
    total = 0;
    for ctype in types:
        print(ctype);
        result_files_prefix = os.path.join(in_dir, ctype, model_prefix);
        lbl = np.load(result_files_prefix + '_individual_labels.npy', allow_pickle=True);
        pred = np.load(result_files_prefix + '_pred_new.npy', allow_pickle=True);
        filenames = pickle.load(open(result_files_prefix + '_filename.pkl', "rb"));
        filenames = np.array(filenames);
        pred = pred.squeeze();
        print('pred[0] = ', pred[0])
        pred_logits= np.log(pred/(1-pred))
        print('pred_logits[0] =', pred_logits[0])
        pred_exp = np.exp(pred_logits)
        print('pred_exp[0] = ', pred_exp[0])
        pred_exp_sum = pred_exp.sum(axis=-1).reshape(-1,1)
        print('pred_exp_sum[0] = ', pred_exp_sum[0])
        pred_sm = pred_exp / pred_exp_sum ;
        print('pred_sm[0] = ', pred_sm[0])
        pred_t1 = pred_sm[:,1];
        print('pred_t1[0] = ', pred_t1[0])
        count = lbl.shape[0];
        P = lbl[np.where(pred_t1 > threshold)].shape[0]
        TP = lbl[np.where(pred_t1 > threshold)].sum();
        FP = P - TP;
        N = count - P;
        FN = lbl[np.where(pred_t1 <= threshold)].sum();
        TN = N - FN;
        lbl_p = lbl[np.where(pred_t1 > threshold)];
        #print(filenames[np.where(lbl_p == 0)]);
        print("TP = ", TP);
        print("FP = ", FP);
        print("TN = ", TN);
        print("FN = ", FN);
        prec = TP/float(TP + FP);
        recall = TP/float(TP + FN);
        print("prec = ", prec);
        print("recall = ", recall);
        f1 = 2/(1/prec + 1/recall);
        accuracy = (TP+TN)/float(count);
        out_file.write(ctype + ',' + "Pred. Positive" + "," + "Pred. Negative" + "," + "Prec."+ "," + "Recall"+ "," + "F1" + "," + "accuracy" +'\r\n');
        out_file.write("Label Positive" + "," + str(TP) + "," + str(FN) + "," + "{:.2f}".format(prec) + "," + "{:.2f}".format(recall) + "," + "{:.2f}".format(f1) + "," + "{:.2f}".format(accuracy) +'\r\n');
        out_file.write("Label Negative" + "," + str(FP) + "," + str(TN) +'\r\n');
        TP_total += TP;
        FP_total += FP;
        TN_total += TN;
        FN_total += FN;
        T_total += TP + TN;
        total += count;
    prec = TP_total/float(TP_total + FP_total);
    recall = TP_total/float(TP_total + FN_total);
    f1 = 2/(1/prec + 1/recall);
    accuracy = (T_total)/float(total);

    out_file.write('\n\n');
    out_file.write('All' + ',' + "Pred. Positive" + "," + "Pred. Negative" + "," + "Prec."+ "," + "Recall"+ "," + "F1" + "," + "accuracy" +'\r\n');
    out_file.write("Label Positive" + "," + str(TP_total) + "," + str(FN_total) + "," + "{:.2f}".format(prec) + "," + "{:.2f}".format(recall) + "," + "{:.2f}".format(f1) + "," + "{:.2f}".format(accuracy) +'\r\n');
    out_file.write("Label Negative" + "," + str(FP_total) + "," + str(TN_total) +'\r\n');

    out_file.close();


def copy_patches_wpred_separate_types(in_dir, out_dir, model_prefix, threshold, types):
    #types = ['coad', 'brca', 'read', 'luad', 'uvm', 'lusc', 'ucec', 'stad', 'blca', 'paad', 'prad', 'cesc', 'skcm'];
    #types = ['uvm'];
    #types = ['ucec'];

    TP_total = 0;
    FP_total = 0;
    TN_total = 0;
    FN_total = 0;
    T_total = 0;
    total = 0;
    for ctype in types:
        print(ctype);
        result_files_prefix = os.path.join(in_dir, ctype, model_prefix);
        lbl = np.load(result_files_prefix + '_individual_labels.npy', allow_pickle=True);
        pred = np.load(result_files_prefix + '_pred_new.npy', allow_pickle=True);
        pred_logits = np.log(pred / (1-pred + 0.0001))
        exp_pred = np.exp(pred_logits - np.max(pred_logits, axis=-1, keepdims=True) + 1)
        exp_pred = exp_pred[..., -1:] / np.sum(exp_pred, axis=-1, keepdims=True)
        #pred_pos = (exp_pred > threshold).squeeze();
        #print(pred_pos.shape)

        filenames = pickle.load(open(result_files_prefix + '_filename.pkl', "rb"));
        filenames = np.array(filenames);
        pred = pred.squeeze();
        pred_t1 = pred[:,1];
        count = lbl.shape[0];
        pred_pos = pred_t1 > threshold;
        lbl_pos = lbl == 1;
        true_pred_files = filenames[np.where(pred_pos == lbl_pos)];
        false_pred_files = filenames[np.where(pred_pos != lbl_pos)];
        print('threshold = ', threshold);
        print('T = ', true_pred_files.shape[0])
        print('F = ', false_pred_files.shape[0])
        print('Accuracy = ', true_pred_files.shape[0] / float(true_pred_files.shape[0] +false_pred_files.shape[0]));
        for i in range(filenames.shape[0]):            
            print("----------------------------------------------");
            file = filenames[i];
            tag = 'F';
            if(pred_pos[i] == lbl_pos[i]):
                tag = 'T';
            print(file);
            file = glob.glob(file.split('.')[0] + '*')[0]           
            print(file);
            print('pred[i] = ', pred[i])
            print('pred_pos[i] = ', pred_pos[i])
            print('exp_pred[i] = ', exp_pred[i])
            print('lbl[i] = ', lbl[i])
            filename, ext = os.path.splitext(os.path.split(file)[1]);            
            dst = os.path.join(out_dir, ctype + '_' + filename + '_'+str(pred[i][0])+'-'+str(pred[i][1])+ '_' + tag + ext);
            print(dst);
            copyfile(file, dst);

        print('threshold = ', threshold);
        print('T = ', true_pred_files.shape[0])
        print('F = ', false_pred_files.shape[0])
        print('Accuracy = ', true_pred_files.shape[0] / float(true_pred_files.shape[0] +false_pred_files.shape[0]));
        #for file in true_pred_files:
        #    print(file);
        #    file = glob.glob(file.split('.')[0] + '*')[0]           
        #    filename, ext = os.path.splitext(os.path.split(file)[1]);            
        #    dst = os.path.join(out_dir, ctype + '_' + filename + '_' + 'T' + ext);
        #    copyfile(file, dst);

        #for file in false_pred_files:
        #    print(file);
        #    file = glob.glob(file.split('.')[0] + '*')[0]          
        #    filename, ext = os.path.splitext(os.path.split(file)[1]);            
        #    dst = os.path.join(out_dir, ctype + '_' + filename + '_' + 'F' + ext);
        #    copyfile(file, dst);


def process_results_auc(in_dir, model_prefix, out_dir):
    print('in process_results_auc')
    sys.stdout.flush();
    out_filename = os.path.join(in_dir, model_prefix + '_auc'+ '.csv');
    out_file = open(out_filename, 'w+');
    #types = ['luad', 'brca', 'acc', 'kirc', 'lihc', 'ov', 'hnsc'];
    #types = ['kirc', 'lihc', 'ov', 'hnsc', 'acc'];

    count = {}
    tp = {}
    fp = {}
    tn = {}
    fn = {}
    #for t in types:
    #    count[t] = 0
    #    tp[t] = 0
    #    fp[t] = 0
    #    tn[t] = 0
    #    fn[t] = 0


    result_files_prefix = os.path.join(in_dir, model_prefix);
    lbl = np.load(result_files_prefix + '_individual_labels.npy', allow_pickle=True);
    if(os.path.isfile(result_files_prefix + '_pred_new.npy')):
        pred = np.load(result_files_prefix + '_pred_new.npy', allow_pickle=True);
    elif(os.path.isfile(result_files_prefix + '_pred_prob.npy')):
        pred = np.load(result_files_prefix + '_pred_prob.npy', allow_pickle=True);    
    pred = pred.squeeze();    
    print('pred.shape = ', pred.shape);
    sys.stdout.flush();
    if(len(pred.shape) > 1 and pred.shape[1]>1):
        pred_t1 = pred[:,1];
    elif(len(pred.shape) > 1 and pred.shape[1]==1):
        pred_t1 = pred[:,0];
    elif(len(pred.shape) == 1):
        pred_t1 = pred;


    print('read pred')
    sys.stdout.flush();
    fpr, tpr, thresholds = metrics.roc_curve(lbl, pred_t1, pos_label=1)
    auc = metrics.auc(fpr, tpr);
    youden_index = tpr-fpr;
    cutoff_youden = thresholds[np.argmax(youden_index)]
    distance = np.sqrt(np.square(1 - tpr) + np.square(fpr));
    cutoff_distance = thresholds[np.argmin(distance)]
    print('calculated auc')
    print('auc', auc)
    print('cutoff_youden', cutoff_youden)
    print('cutoff_distance', cutoff_distance)
    sys.stdout.flush();

    #filenames = pickle.load(open(result_files_prefix + '_filename.pkl', "rb"));
    #filenames = np.array(filenames)

    #threshold = cutoff_youden
    #out_filename = os.path.join(in_dir, model_prefix + '_stats_th-'+ str(threshold)+'_auc-'+str(auc)+'.csv');
    #out_file = open(out_filename, 'w+');

    #for i in range(len(filenames)):
    #    for key in types:
    #        print(key)
    #        print(filenames[i])
    #        if key in filenames[i]:
    #            count[key] += 1 
    #            lbl_val = lbl[i] == 1
    #            pred_val = (pred_t1[i]>= threshold)
    #            if(lbl_val):
    #                if(pred_val):
    #                    tp[key] += 1
    #                else:
    #                    fn[key] += 1
    #            else:
    #                if(pred_val):
    #                    fp[key] += 1
    #                else:
    #                    tn[key] += 1
        
    #out_file.write('dataset,count,tp,fn,fp,tn\n')        
    #for key in types:
    #    out_file.write(str(key)+','+str(count[key])+','+str(tp[key])+','+str(fn[key])+','+str(fp[key])+','+str(tn[key]) +','+'\n')        
    #out_file.close()

    #print('create file1')
    #sys.stdout.flush();

    #threshold = 0.5
    #out_filename = os.path.join(in_dir, model_prefix +'_stats_th-'+ str(threshold)+'.csv');
    #out_file = open(out_filename, 'w+');

    #count = {'luad':0, 'brca':0, 'acc':0, 'kirc':0, 'lihc':0, 'ov':0, 'hnsc':0};
    #tp = {'luad':0, 'brca':0, 'acc':0, 'kirc':0, 'lihc':0, 'ov':0, 'hnsc':0};
    #fp = {'luad':0, 'brca':0, 'acc':0, 'kirc':0, 'lihc':0, 'ov':0, 'hnsc':0};
    #tn = {'luad':0, 'brca':0, 'acc':0, 'kirc':0, 'lihc':0, 'ov':0, 'hnsc':0};
    #fn = {'luad':0, 'brca':0, 'acc':0, 'kirc':0, 'lihc':0, 'ov':0, 'hnsc':0};

    #for i in range(len(filenames)):
    #    for key in types:
    #        if key in filenames[i]:
    #            count[key] += 1 
    #            lbl_val = lbl[i] == 1
    #            pred_val = (pred_t1[i]>= threshold)
    #            if(lbl_val):
    #                if(pred_val):
    #                    tp[key] += 1
    #                else:
    #                    fn[key] += 1
    #            else:
    #                if(pred_val):
    #                    fp[key] += 1
    #                else:
    #                    tn[key] += 1
        
    #out_file.write('dataset,count,tp,fn,fp,tn\n')        
    #for key in types:
    #    out_file.write(str(key) +','+str(count[key])+','+str(tp[key])+','+str(fn[key])+','+str(fp[key])+','+str(tn[key]) +','+'\n')        
    #out_file.close()
    #print('create file2')
    #sys.stdout.flush();


def process_results_thresh(in_dir, model_prefix, out_dir,threshold, types):
    print('in process_results_thresh')
    sys.stdout.flush();
    #types = ['luad', 'brca', 'acc', 'kirc', 'lihc', 'ov', 'hnsc'];

    #types = ['coad', 'read', 'uvm', 'lusc', 'ucec', 'stad', 'blca', 'paad', 'prad', 'cesc', 'skcm', 'brca_additional+', 'luad_additional+'];


    count = {}
    tp = {}
    fp = {}
    tn = {}
    fn = {}
    for t in types:
        count[t] = 0
        tp[t] = 0
        fp[t] = 0
        tn[t] = 0
        fn[t] = 0

    result_files_prefix = os.path.join(in_dir, model_prefix);
    lbl = np.load(result_files_prefix + '_individual_labels.npy', allow_pickle=True);
    if(os.path.isfile(result_files_prefix + '_pred_new.npy')):
        pred = np.load(result_files_prefix + '_pred_new.npy', allow_pickle=True);
    elif(os.path.isfile(result_files_prefix + '_pred_prob.npy')):
        pred = np.load(result_files_prefix + '_pred_prob.npy', allow_pickle=True);    
    pred = pred.squeeze();    
    print('pred.shape = ', pred.shape);
    sys.stdout.flush();
    if(len(pred.shape) > 1 and pred.shape[1]>1):
        pred_t1 = pred[:,1];
    elif(len(pred.shape) > 1 and pred.shape[1]==1):
        pred_t1 = pred[:,0];
    elif(len(pred.shape) == 1):
        pred_t1 = pred;
    ctype = []



    if(os.path.isfile(result_files_prefix + '_filename.npy')):
        filenames = np.load(open(result_files_prefix + '_filename.npy', "rb"), allow_pickle=True);
    else:
        filenames = pickle.load(open(result_files_prefix + '_filename.pkl', "rb"));
    filenames = np.array(filenames)

    out_filename = os.path.join(in_dir, model_prefix + '_stats_th-'+ str(threshold)+'.csv');
    out_file = open(out_filename, 'w+');
    print('out_filename',out_filename)

    for i in range(len(filenames)):
        found = False
        for key in types:
            print(key)
            print(filenames[i])
            if key in filenames[i]:
                count[key] += 1 
                lbl_val = lbl[i] == 1
                #pred_val = (pred_t1[i]>= threshold)
                pred_val = (pred_t1[i]> threshold)
                if(lbl_val):
                    if(pred_val):
                        tp[key] += 1
                    else:
                        fn[key] += 1
                else:
                    if(pred_val):
                        fp[key] += 1
                    else:
                        tn[key] += 1
                ctype.append(key)
                found = True

        if(not found):
            ctype.append('')

    ctype = np.array(ctype)
    out_file.write('dataset,count,tp,fn,fp,tn, precision, recall, f-score, auc, accuracy\n')        
    for key in types:
        fpr, tpr, thresholds = metrics.roc_curve(lbl[ctype==key], pred_t1[ctype==key], pos_label=1)
        auc = metrics.auc(fpr, tpr);
        if(tp[key] + fp[key] == 0):
            precision = 1
        else:
            precision = tp[key] /(tp[key] +fp[key] )
        if(tp[key] + fn[key] == 0):
            recall = 1
        else:
            recall = tp[key] /(tp[key] +fn[key] )
        fscore = 2*precision*recall/(precision+recall)
        accuracy = (tp[key] + tn[key])/(count[key] )
        out_file.write(str(key)+','+str(count[key])+','+str(tp[key])+','+str(fn[key])+','+str(fp[key])+','+str(tn[key]) +','+str(precision) +','+str(recall) +','+str(fscore) +','+str(auc) +','+str(accuracy) +','+'\n')        

    out_file.close()

    print('create file1')
    sys.stdout.flush();

    


def process_results(in_dir, out_dir, model_prefix, dataset_name, threshold):
    out_filename = os.path.join(in_dir, model_prefix + '_'+dataset_name+'_stats2_th-'+ str(threshold)+'.csv');
    out_file = open(out_filename, 'w+');

    out_file.write(model_prefix+'\n\n');
    result_files_prefix = os.path.join(in_dir, model_prefix);
    lbl = np.load(result_files_prefix + '_individual_labels.npy', allow_pickle=True);
    if(os.path.isfile(result_files_prefix + '_pred_new.npy')):
        pred = np.load(result_files_prefix + '_pred_new.npy', allow_pickle=True);
    elif(os.path.isfile(result_files_prefix + '_pred_prob.npy')):
        pred = np.load(result_files_prefix + '_pred_prob.npy', allow_pickle=True);    
    pred = pred.squeeze();    
    print('pred.shape = ', pred.shape);
    if(len(pred.shape) > 1 and pred.shape[1]>1):
        pred_t1 = pred[:,1];
    elif(len(pred.shape) > 1 and pred.shape[1]==1):
        pred_t1 = pred[:,0];
    elif(len(pred.shape) == 1):
        pred_t1 = pred;
    count = lbl.shape[0];
    P = lbl[np.where(pred_t1 > threshold)].shape[0]
    TP = lbl[np.where(pred_t1 > threshold)].sum();
    FP = P - TP;
    N = count - P;
    FN = lbl[np.where(pred_t1 <= threshold)].sum();
    TN = N - FN;
    prec = TP/float(TP + FP);
    recall = TP/float(TP + FN);
    f1 = 2/(1/prec + 1/recall);
    accuracy = (TP+TN)/float(count);

    fpr, tpr, thresholds = metrics.roc_curve(lbl, pred_t1, pos_label=1)
    auc = metrics.auc(fpr, tpr);
    youden_index = tpr-fpr;
    cutoff_youden = thresholds[np.argmax(youden_index)]
    distance = np.sqrt(np.square(1 - tpr) + np.square(fpr));
    cutoff_distance = thresholds[np.argmin(distance)]

    out_file.write(dataset_name + ',' + "Pred. Pos." + "," + "Pred. Neg." + "," + "Prec."+ "," + "Recall" + "," + "F1" + "," + "Accuracy" + "," + "AUC" +'\r\n');
    out_file.write("Label Pos." + "," + str(TP) + "," + str(FN) + "," + "{:.2f}".format(prec) + "," + "{:.2f}".format(recall)  + "," + "{:.2f}".format(f1) + "," + "{:.4f}".format(accuracy)+ "," + "{:.4f}".format(auc) +'\r\n');
    out_file.write("Label Neg." + "," + str(FP) + "," + str(TN) +'\r\n');


    out_file.close();

    print('auc = ', auc);
    print('cutoff_youden = ', cutoff_youden);
    print('cutoff_distance = ', cutoff_distance);

    #for i in range(thresholds.shape[0]):
    #    print('fpr = ', fpr[i], 'tpr = ', tpr[i], 'thresholds = ', thresholds[i]);

    #plt.plot(fpr, tpr);
    #plt.show();


def process_results_separate_types_no_write(in_dir, out_dir, model_prefix, threshold, types):
    #types = ['coad', 'brca', 'read', 'luad', 'uvm', 'lusc', 'ucec', 'stad', 'blca', 'paad', 'prad', 'cesc', 'skcm'];
    #types = ['ucec'];

    TP_total = 0;
    FP_total = 0;
    TN_total = 0;
    FN_total = 0;
    for ctype in types:
        print(ctype);
        result_files_prefix = os.path.join(in_dir, ctype, model_prefix);
        lbl = np.load(result_files_prefix + '_individual_labels.npy', allow_pickle=True);
        pred = np.load(result_files_prefix + '_pred_new.npy', allow_pickle=True);
        filenames = pickle.load(open(result_files_prefix + '_filename.pkl', "rb"));
        filenames = np.array(filenames);
        pred = pred.squeeze();
        pred_t1 = pred[:,1];
        count = lbl.shape[0];
        P = lbl[np.where(pred_t1 > threshold)].shape[0]
        TP = lbl[np.where(pred_t1 > threshold)].sum();
        FP = P - TP;
        N = count - P;
        FN = lbl[np.where(pred_t1 <= threshold)].sum();
        TN = N - FN;
        lbl_p = lbl[np.where(pred_t1 > threshold)];
        filenames_p = filenames[np.where(pred_t1 > threshold)];
        pred_t1_p = pred_t1[np.where(pred_t1 > threshold)];
        print(filenames_p[np.where(lbl_p == 0)]);
        print(pred_t1_p[np.where(lbl_p == 0)]);
        print("TP = ", TP);
        print("FP = ", FP);
        print("TN = ", TN);
        print("FN = ", FN);
        prec = TP/float(TP + FP);
        recall = TP/float(TP + FN);
        print("prec = ", prec);
        print("recall = ", recall);
        f1 = 2/(1/prec + 1/recall);
        TP_total += TP;
        FP_total += FP;
        TN_total += TN;
        FN_total += FN;

    prec = TP_total/float(TP_total + FP_total);
    recall = TP_total/float(TP_total + FN_total);
    f1 = 2/(1/prec + 1/recall);

def process_results_separate_types_auc(in_dir, out_dir, model_prefix, types):
    #types = ['coad', 'brca', 'read', 'luad', 'uvm', 'lusc', 'ucec', 'stad', 'blca', 'paad', 'prad', 'cesc', 'skcm'];
    #types = ['coad', 'brca', 'read', 'luad', 'lusc', 'ucec', 'stad', 'blca', 'paad', 'prad', 'cesc', 'skcm'];
    #types = ['kirc', 'lihc', 'ov', 'hnsc', 'acc'];
    #types = ['luad'];
    #out_filename = os.path.join(in_dir, model_prefix + '_ctypes_stats_th-'+ str(threshold)+'.csv');
    #out_file = open(out_filename, 'w+');

    out_filename = os.path.join(in_dir, model_prefix + '_stats_th-'+ str(threshold)+'.csv');
    out_file = open(out_filename, 'w+');
    #out_file.write(model_prefix+'\n\n');
    TP_total = 0;
    FP_total = 0;
    TN_total = 0;
    FN_total = 0;
    T_total = 0;
    total = 0;
    pred_all_t1 = None;
    lbl_all = None;
    for ctype in types:
        print(ctype);
        result_files_prefix = os.path.join(in_dir, ctype, model_prefix);
        lbl = np.load(result_files_prefix + '_individual_labels.npy', allow_pickle=True);
        #pred = np.load(result_files_prefix + '_pred_new.npy');
        if(os.path.isfile(result_files_prefix + '_pred_new.npy')):
            pred = np.load(result_files_prefix + '_pred_new.npy', allow_pickle=True);
        elif(os.path.isfile(result_files_prefix + '_pred_prob.npy')):
            pred = np.load(result_files_prefix + '_pred_prob.npy', allow_pickle=True);    
        filenames = pickle.load(open(result_files_prefix + '_filename.pkl', "rb"));
        filenames = np.array(filenames);
        pred = pred.squeeze();
        #pred_t1 = pred[:,1];
        if(len(pred.shape) > 1 and pred.shape[1]>1):
            pred_t1 = pred[:,1];
        elif(len(pred.shape) > 1 and pred.shape[1]==1):
            pred_t1 = pred[:,0];
        elif(len(pred.shape) == 1):
            pred_t1 = pred;
        if(pred_all_t1 is None):
            pred_all_t1 = pred_t1;
            lbl_all = lbl;
        else:
            pred_all_t1 = np.concatenate((pred_all_t1, pred_t1), axis=-1);
            lbl_all = np.concatenate((lbl_all,lbl), axis=-1);
        fpr, tpr, thresholds = metrics.roc_curve(lbl, pred_t1, pos_label=1)
        auc = metrics.auc(fpr, tpr);
        print('auc = ', auc);
        print(' ');
        youden_index = tpr-fpr;
        cutoff = thresholds[np.argmax(youden_index)]
        print('cutoff = ', cutoff);


    fpr, tpr, thresholds = metrics.roc_curve(lbl_all, pred_all_t1, pos_label=1)
    auc = metrics.auc(fpr, tpr);
    youden_index = tpr-fpr;
    cutoff = thresholds[np.argmax(youden_index)]

    #for i in range(thresholds.shape[0]):
    #    print('fpr = ', fpr[i], 'tpr = ', tpr[i], 'thresholds = ', thresholds[i]);
    print('all auc = ', auc);
    print('all cutoff = ', cutoff);

    #plt.plot(fpr, tpr);
    #plt.show();

    #out_file.write('\n\n');
    #out_file.write('All' + ',' + "Pred. Positive" + "," + "Pred. Negative" + "," + "Prec."+ "," + "Recall"+ "," + "F1" + "," + "accuracy" +'\r\n');
    #out_file.write("Label Positive" + "," + str(TP_total) + "," + str(FN_total) + "," + "{:.2f}".format(prec) + "," + "{:.2f}".format(recall) + "," + "{:.2f}".format(f1) + "," + "{:.2f}".format(accuracy) +'\r\n');
    #out_file.write("Label Negative" + "," + str(FP_total) + "," + str(TN_total) +'\r\n');

    #out_file.close();

    #fig,ax = plt.subplots(1)
    #ax.plot(fpr, tpr);
    #fig.savefig('test.png');


def process_results_separate_types_roc_curve(in_dir, model_prefix_list, label_list, types):
    #types = ['coad', 'brca', 'read', 'luad', 'uvm', 'lusc', 'ucec', 'stad', 'blca', 'paad', 'prad', 'cesc', 'skcm'];

    fig,ax = plt.subplots(1)
    for i in range(len(model_prefix_list)):
        model_prefix = model_prefix_list[i];
        label = label_list[i];
        print(model_prefix);
        TP_total = 0;
        FP_total = 0;
        TN_total = 0;
        FN_total = 0;
        T_total = 0;
        total = 0;
        pred_all_t1 = None;
        lbl_all = None;
        for ctype in types:
            print(ctype);
            result_files_prefix = os.path.join(in_dir, ctype, model_prefix);
            lbl = np.load(result_files_prefix + '_individual_labels.npy', allow_pickle=True);
            pred = np.load(result_files_prefix + '_pred_new.npy', allow_pickle=True);
            filenames = pickle.load(open(result_files_prefix + '_filename.pkl', "rb"));
            filenames = np.array(filenames);
            pred = pred.squeeze();
            pred_t1 = pred[:,1];
            if(pred_all_t1 is None):
                pred_all_t1 = pred_t1;
                lbl_all = lbl;
            else:
                pred_all_t1 = np.concatenate((pred_all_t1, pred_t1), axis=-1);
                lbl_all = np.concatenate((lbl_all,lbl), axis=-1);
        

        fpr, tpr, thresholds = metrics.roc_curve(lbl_all, pred_all_t1, pos_label=1)

        ax.plot(fpr, tpr, label=label);

    ax.legend();
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate');
    plt.title('ROC');
    fig.savefig('test.png');

def generate_result_files_all_types_for_old_model(in_dir, out_dir, model_prefix, types):
    #types = ['coad', 'brca', 'read', 'luad', 'uvm', 'lusc', 'ucec', 'stad', 'blca', 'paad', 'prad', 'cesc', 'skcm'];
    #types = ['coad', 'brca', 'read', 'luad', 'lusc', 'ucec', 'stad', 'blca', 'paad', 'prad', 'cesc', 'skcm'];

    for ctype in types:
        print(ctype);
        files_prefix = os.path.join(in_dir, ctype, '**', '*.png');
        result_files_prefix = os.path.join(out_dir, ctype, model_prefix);
        files = glob.glob(files_prefix, recursive=True);
        labels = [];
        pred = [];
        pred_manual_thesh = [];
        for file in files:
            filename = os.path.split(file)[1];
            label_str = filename[-5];
            label = int(label_str);
            labels.append(label);
            filename_split = filename[0:-6].split('-');
            pred_manual_thresh_str = filename_split[-1];
            pred_prob_str = filename_split[-3];
            pred_manual_thesh.append(np.array([0, int(pred_manual_thresh_str)]));
            pred.append(np.array([0, float(pred_prob_str)]));
            print(file)
            print(label)
            print(float(pred_prob_str))
            print(int(pred_manual_thresh_str))

        labels_arr = np.array(labels);
        labels_arr.dump(result_files_prefix +'1'+ '_individual_labels.npy')
        labels_arr.dump(result_files_prefix +'2'+'_individual_labels.npy')
        pickle.dump(files, open(result_files_prefix +'1'+ '_filename.pkl', 'wb'))
        pickle.dump(files, open(result_files_prefix +'2'+ '_filename.pkl', 'wb'))
        pred_prob_arr = np.array(pred);
        pred_prob_arr.dump(result_files_prefix +'2'+ '_pred_new.npy')
        pred_manual_thresh_arr = np.array(pred_manual_thesh);
        pred_manual_thresh_arr.dump(result_files_prefix + '1'+'_pred_new.npy')

        
def generate_result_files_for_old_model(in_dir, out_dir, model_prefix):

    files_prefix = os.path.join(in_dir,'**', '*.png');
    result_files_prefix = os.path.join(out_dir, model_prefix);
    files = glob.glob(files_prefix, recursive=True);
    labels = [];
    pred = [];
    pred_manual_thesh = [];
    for file in files:
        filename = os.path.split(file)[1];
        label_str = filename[-5];
        label = int(label_str);
        labels.append(label);
        filename_split = filename[0:-6].split('-');
        pred_manual_thresh_str = filename_split[-1];
        pred_prob_str = filename_split[-3];
        pred_manual_thesh.append(np.array([0, int(pred_manual_thresh_str)]));
        pred.append(np.array([0, float(pred_prob_str)]));
        print(file)
        print(label)
        print(float(pred_prob_str))
        print(int(pred_manual_thresh_str))

    labels_arr = np.array(labels);
    labels_arr.dump(result_files_prefix +'1'+ '_individual_labels.npy')
    labels_arr.dump(result_files_prefix +'2'+'_individual_labels.npy')
    pickle.dump(files, open(result_files_prefix +'1'+ '_filename.pkl', 'wb'))
    pickle.dump(files, open(result_files_prefix +'2'+ '_filename.pkl', 'wb'))
    pred_prob_arr = np.array(pred);
    pred_prob_arr.dump(result_files_prefix +'2'+ '_pred_new.npy')
    pred_manual_thresh_arr = np.array(pred_manual_thesh);
    pred_manual_thresh_arr.dump(result_files_prefix + '1'+'_pred_new.npy')

def generate_labels_files_all_types(dir, model_prefix, types):
    #types = ['coad', 'brca', 'read', 'luad', 'uvm', 'lusc', 'ucec', 'stad', 'blca', 'paad', 'prad', 'cesc', 'skcm'];

    for ctype in types:
        print(ctype);

        result_files_prefix = os.path.join(dir, ctype, model_prefix);
        lbl = np.load(result_files_prefix + '_individual_labels.npy', allow_pickle=True);        
        filenames_old = pickle.load(open(result_files_prefix + '_filename.pkl', "rb"));
        filenames_old = np.array(filenames_old);

        filenames_new = [];
        labels_new = [];

        for file_old in filenames_old:
            print(file_old);
            file_pattern = file_old[:-40]  + '*.png';
            file_new = glob.glob(file_pattern)[0];
            print(file_new);
            filenames_new.append(file_new);
            label_new_str = file_new[-5];
            label_new = int(label_new_str);
            labels_new.append(label_new);            

        labels_arr = np.array(labels_new);
        labels_arr.dump(result_files_prefix + '_individual_labels.npy')
        pickle.dump(filenames_new, open(result_files_prefix + '_filename.pkl', 'wb'))

if __name__ == "__main__":
    in_dir = "/home/shahira/TIL_classification/eval_val_old_plus_new2/";
    #types = ['luad', 'kirc', 'lihc', 'ov', 'hnsc', 'acc', 'brca'];
    #types = ['coad', 'read', 'uvm', 'lusc', 'ucec', 'stad', 'blca', 'paad', 'prad', 'cesc', 'skcm', 'brca_additional+', 'luad_additional+'];
    #types = ['sarc', 'ov'];
    types = ['kirc', 'lihc', 'ov', 'hnsc', 'acc'];

    out_dir= in_dir;

    model_prefix = "tcga_vgg16_mix_new3"
    threshold = 0.4
    model_prefix = "tcga_incv4_mix_new3"
    threshold = 0.41
    model_prefix = "resnet34"
    threshold = 0.56
    process_results_auc(in_dir, model_prefix,out_dir);
    #process_results_thresh(in_dir, model_prefix,out_dir,threshold,types);
    #process_results_separate_types(in_dir, out_dir, model_prefix, threshold, types)

