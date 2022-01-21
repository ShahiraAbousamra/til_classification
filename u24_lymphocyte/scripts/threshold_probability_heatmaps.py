import numpy as np;
#import pickle;
import os;
import sys;
import glob;
from shutil import copyfile
import json;
import configparser;

def edit_json(json_folder_path, out_dir, threshold, suffix = None, new_heatmap_version_name= None):
    heatmap_files_pattern = 'heatmap_*.json'
    meta_files_pattern = 'meta_*.json'

    if(suffix is None and new_heatmap_version_name is None):
        return False;

    ## Create the output folder
    #if(new_heatmap_version_name is None):
    #    folder_name = json_folder_path.split('/')[-1] + suffix;
    #else:
    #    folder_name = new_heatmap_version_name;
    #dest_dir = os.path.join(out_dir, folder_name );
    dest_dir = out_dir;

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir);

  # edit the probability and version name in heatmap json files to be 0.0 or 1.0 based on threshold
    files = glob.glob(os.path.join(json_folder_path, heatmap_files_pattern), recursive=True);
    for filepath in files:
        print(filepath);
        file = open(filepath, 'r');
        filename = os.path.split(filepath)[1];
        dest_filepath = os.path.join(dest_dir, filename);

        # skip is file already exists
        if(os.path.isfile(dest_filepath )): 
            continue;    

        dest_file = open(dest_filepath, 'w');
       
        for line in file:
            line_json = json.loads(line);
            #print(line_json);
            #print(line_json['properties']['multiheat_param']['metric_array'][0])
            prob = line_json['properties']['metric_value'];
            prob_new = 0.0;
            if(prob >= threshold):
                prob_new = 1.0;
            line_json['properties']['metric_value'] = prob_new;
            line_json['properties']['multiheat_param']['metric_array'][0] = prob_new;

            heatmap_version = line_json['provenance']['analysis']['execution_id'];
            indx = heatmap_version.rfind('-');
            #heatmap_version_new = heatmap_version[0:heatmap_version.rfind('-')] + '_thresh_'+str(threshold) + heatmap_version[indx:];
            if(new_heatmap_version_name is None):
                heatmap_version_new = heatmap_version[0:heatmap_version.rfind('-')] + suffix + heatmap_version[indx:];
            else:
                heatmap_version_new = new_heatmap_version_name + heatmap_version[indx:];
            line_json['provenance']['analysis']['execution_id'] = heatmap_version_new ;

            line_new = json.dumps(line_json);
            #print(line_new);
            dest_file.write(line_new);
            dest_file.write('\n')
        file.close();
        dest_file.close();

    # edit the version name in heatmap json files 
    files = glob.glob(os.path.join(json_folder_path, meta_files_pattern), recursive=True);
    for filepath in files:
        print(filepath);
        file = open(filepath, 'r');
        filename = os.path.split(filepath)[1];
        dest_filepath = os.path.join(dest_dir, filename);

        # skip is file already exists
        if(os.path.isfile(dest_filepath )): 
            continue;    

        dest_file = open(dest_filepath, 'w');
       
        for line in file:
            line_json = json.loads(line);

            heatmap_version = line_json['provenance']['analysis_execution_id'];
            indx = heatmap_version.rfind('-');
            #heatmap_version_new = heatmap_version[0:heatmap_version.rfind('-')] + '_thresh_'+str(threshold) + heatmap_version[indx:];
            if(new_heatmap_version_name is None):
                heatmap_version_new = heatmap_version[0:heatmap_version.rfind('-')] + suffix + heatmap_version[indx:];
            else:
                heatmap_version_new = new_heatmap_version_name + heatmap_version[indx:];

            line_json['provenance']['analysis_execution_id'] = heatmap_version_new ;

            title = line_json['title']
            indx = title.rfind('-');
            #title_new = title[0:title.rfind('-')] + '_thresh_'+str(threshold) + title[indx:];
            if(new_heatmap_version_name is None):
                title_new = title[0:title.rfind('-')] + suffix + title[indx:];
            else:
                title_new = new_heatmap_version_name + title[indx:];
            line_json['title'] = title_new ;

            line_new = json.dumps(line_json);
            dest_file.write(line_new);
            dest_file.write('\n')
        file.close();
        dest_file.close();


def edit_prediction_txt(heatmap_txt_folder_path, out_dir, threshold, suffix = None, new_heatmap_version_name= None):
    prediction_files_pattern = 'prediction-*'
    color_files_pattern = 'color-*'

    if(suffix is None and new_heatmap_version_name is None):
        return False;

    ## Create the output folder
    #if(new_heatmap_version_name is None):
    #    folder_name = heatmap_txt_folder_path.split('/')[-1] + suffix;
    #else:
    #    folder_name = new_heatmap_version_name;
    #dest_dir = os.path.join(out_dir, folder_name );
    dest_dir = out_dir;

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir);

  # edit the probability and version name in heatmap json files to be 0.0 or 1.0 based on threshold
    files = glob.glob(os.path.join(heatmap_txt_folder_path, prediction_files_pattern), recursive=True);
    delimeter = ' ';
    new_line_char = '\n';
    for filepath in files:
        print(filepath);
        file = open(filepath, 'r');
        filename = os.path.split(filepath)[1];
        dest_filepath = os.path.join(dest_dir, filename);

        # skip is file already exists
        if(os.path.isfile(dest_filepath )): 
            continue;    

        dest_file = open(dest_filepath, 'w');
       
        #counter = 0;
        for line in file:
            #print(line);
            fields = line.split();
            #print(fields);
            prob = float(fields[2]);
            prob_new = 0.0;
            if(prob > threshold):
                prob_new = 1.0;
            #print('prob_old = ', prob, ' prob_new = ', prob_new)
            line_new = fields[0] + delimeter + fields[1] + delimeter + str(prob_new) + delimeter + fields[3] + new_line_char;
            #print(line_new);
            dest_file.write(line_new);
 
        file.close();
        dest_file.close();

    # copy the color files to destination
    files = glob.glob(os.path.join(heatmap_txt_folder_path, color_files_pattern), recursive=True);
    for filepath in files:
        filename = os.path.split(filepath)[1];
        dest_filepath = os.path.join(dest_dir, filename);
        copyfile(filepath, dest_filepath);

def get_threshold_from_config(config_filepath):
    # read the config file
    config = configparser.ConfigParser();

    config.read(config_filepath);  
    threshold = float(config['TESTER']['threshold'].strip()); 
    return threshold;

if __name__ == "__main__":

    in_dir_json = sys.argv[1];
    out_dir_json = sys.argv[2];
    in_dir_txt = sys.argv[3];
    out_dir_txt = sys.argv[4];
    config_filepath = sys.argv[5];
    new_heatmap_version_name = sys.argv[6];

    threshold = get_threshold_from_config(config_filepath);

    edit_json(in_dir_json, out_dir_json, threshold, new_heatmap_version_name = new_heatmap_version_name);
    edit_prediction_txt(in_dir_txt, out_dir_txt, threshold, new_heatmap_version_name = new_heatmap_version_name);



    
