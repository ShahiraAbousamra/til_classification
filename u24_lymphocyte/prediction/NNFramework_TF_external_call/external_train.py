import numpy as np
import sys
sys.path.append("..");
sys.path.append(".");
sys.path.append("../..");
sys.path.append("...");
from NNFramework_TF.sa_runners import tf_classifier_runner;



if __name__ == "__main__":
    
    ## LUAD   ###############################################################  
    ## LUAD - semiauto - InceptionV4
    #config_filepath = "/home/shahira/NNFramework_TF_model_config/config_tcga_incv4_b128_crop100_noBN_wd5e-4_d75_luad_semiauto.ini";
    ## LUAD - semiauto - VGG16
    #config_filepath = "/home/shahira/NNFramework_TF_model_config/config_tcga_vgg16_b128_crop100_luad_semiauto.ini";
    ## LUAD - manual - InceptionV4
    #config_filepath = "/home/shahira/NNFramework_TF_model_config/config_tcga_incv4_b128_crop100_noBN_wd5e-4_d75_luad_manual.ini";
    ## LUAD - manual - VGG16
    #config_filepath = "/home/shahira/NNFramework_TF_model_config/config_tcga_vgg16_b128_crop100_luad_manual.ini";

    ## SKCM   ############################################################### 
    ## SKCM - semiauto - InceptionV4
    #config_filepath = "/home/shahira/NNFramework_TF_model_config/config_tcga_incv4_b128_crop100_noBN_wd5e-4_d75_skcm_semiauto.ini";
    ## SKCM - semiauto - VGG16
    #config_filepath = "/home/shahira/NNFramework_TF_model_config/config_tcga_vgg16_b128_crop100_skcm_semiauto.ini";
    ## SKCM - manual - InceptionV4
    #config_filepath = "/home/shahira/NNFramework_TF_model_config/config_tcga_incv4_b128_crop100_noBN_wd5e-4_d75_skcm_manual.ini";
    ## SKCM - manual - VGG16
    #config_filepath = "/home/shahira/NNFramework_TF_model_config/config_tcga_vgg16_b128_crop100_skcm_manual.ini";

    #config_filepath = "/home/shahira/NNFramework_TF/config/config_tcga_incv4_b128_crop100_noBN_wd5e-4_d75_HEaug_luad_semiauto.ini";

    tf_classifier_runner.main(sys.argv[1:])