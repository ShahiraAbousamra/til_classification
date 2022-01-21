import numpy as np
import sys
sys.path.append("..");
sys.path.append(".");
sys.path.append("../..");
sys.path.append("...");
#try:
#    from NNFramework_TF.sa_runners.tf_classifier_runner_external_input import load_model;
#except:
#    from pytorch_resnet34_model.load_til_resnet34_external_input import load_model


def load_external_model(model_path):
    # Load your model here
    if('resnet' in model_path):
        from pytorch_resnet34_model.load_til_resnet34_external_input import load_model
    else:
        from NNFramework_TF.sa_runners.tf_classifier_runner_external_input import load_model;
    model = load_model(model_path)
    return model

def pred_by_external_model(model, inputs):
    # Get prediction here
    # model:
    #     A model loaded by load_external_model
    # inputs :
    #     float32 numpy array with shape N x 3 x 100 x 100
    #     Range of value: 0.0 ~ 255.0
    #     You may need to rearrange inputs:
    #     inputs = inputs.transpose((0, 2, 3, 1))
    # Expected output:
    #     float32 numpy array with shape N x 1
    #     Each entry is a probability ranges 0.0 ~ 1.0
    inputs = inputs.transpose((0, 2, 3, 1))
    pred = model.predict(inputs)
    #exp_pred = np.exp(pred - np.max(pred, axis=-1, keepdims=True) + 1)
    #return exp_pred[..., -1:] / np.sum(exp_pred, axis=-1, keepdims=True)

    #sig_pred = (1 / (1 + np.exp(-pred)))
    #sig_pred = sig_pred[..., -1:] ;
    #return sig_pred

    return pred;

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
    ## Resnet34
    #config_filepath = "/gpfs/projects/KurcGroup/sabousamra/TIL_classification/models/config_resnet-mix-new3_test_ext.ini";
    # VGG16
    config_filepath = "/gpfs/projects/KurcGroup/sabousamra/TIL_classification/models/NNFramework_TF_models/config_vgg-mix-new3_test_ext.ini";

    print(config_filepath)
    model = load_external_model(config_filepath);
    print('load_external_model called')
    inputs = np.random.rand(10, 3, 200, 200);
    print('inputs.shape',inputs.shape)
    import skimage.io as io
    im = io.imread('/gpfs/projects/KurcGroup/sabousamra/TIL_classification/datasets/superpatches_anno/superpatches_all/TCGA-L5-A4OS-01Z-00-DX1.DB97C677-73CB-49E9-8F0A-216EFFF364EC_48672_29203_61_58.png')
    im = im[:,:,:3]
    print('im.shape',im.shape)
    im0 = im[400:600, 400:600, :].transpose((2,0,1))
    print('im0.shape',im0.shape)
    im0 = im0[np.newaxis,:]
    print('im0.max()',im0.max())
    
    print('im0.shape',im0.shape)
    inputs[0] = im0
    print('inputs created')
    print('inputs.shape',inputs.shape)
    pred = pred_by_external_model(model, inputs)
    print('after predict')
    print(pred);
