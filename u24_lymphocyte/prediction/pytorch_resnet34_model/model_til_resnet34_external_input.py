import torch.nn as nn
import torch.nn.functional as F
import torch
#from torchvision import models
import collections
from distutils.util import strtobool;
import cv2
import numpy as np
from torchvision import transforms



class TILClassifierExternalInputModel():
    def __init__(self, model, device, is_binary_output=False, threshold=0.5, max_side=100, resize_side=100):
        super(TILClassifierExternalInputModel,self).__init__()


        self.device = device
        self.model = model
        self.is_binary_output = is_binary_output
        self.threshold = threshold
        self.max_side = max_side
        self.resize_side = resize_side
        
        self.sig_layer = torch.nn.Sigmoid()



    def predict(self,x):
        if(len(x.shape)  > 4):
            x = x.squeeze()
        x_tensor = self.preprocess_input(x)
        x_tensor=x_tensor.to(self.device)
        y = self.model(x_tensor)
        y = self.sig_layer(y).detach().cpu().numpy().squeeze()
        if(self.is_binary_output):
            y = (y > self.threshold).astype(np.float);
        y = y[:, np.newaxis]
        return y

    def preprocess_input(self, inputs):
        #print('inputs.shape a',inputs.shape)
        #np.clip(inputs, 0, 255, inputs);
        #print('inputs.shape c',inputs.shape)
        inputs /= 255;
        #if(self.max_side > 0):
        #    h = inputs.shape[-3]
        #    w = inputs.shape[-2]
        #    h2 = h
        #    w2 = w
        #    crop = False
        #    if(h > self.max_side):
        #        h2 = self.max_side
        #        crop = True
        #    if(w > self.max_side):
        #        w2 = self.max_side
        #        crop = True
        #    if(crop):
        #        print('cropping')
        #        y = (h - h2)// 2
        #        x = (w - w2)// 2
        #        #y=0
        #        #x=0
        #        #if(not (h2 ==h)):
        #        #    y = np.random.randint(0, high = h-h2)
        #        #if(not (w2 ==w)):
        #        #    x = np.random.randint(0, high = w-w2)
        #        inputs = inputs[:,y:y+h2, x:x+w2, :]
        #        #gt_dmap = gt_dmap[y:y+h2, x:x+w2]

        #if(self.resize_side > 0 and (inputs.shape[1] != self.resize_side or inputs.shape[2] != self.resize_side )):
        if(self.resize_side > 0 ):
            #print('resizing')
            inputs2 = np.zeros((inputs.shape[0], self.resize_side,self.resize_side, inputs.shape[3]))
            for i in range(inputs.shape[0]):
                img = cv2.resize(inputs[i].squeeze(), (self.resize_side,self.resize_side))
                inputs2[i] = img
            inputs = inputs2
        inputs=inputs.transpose((0,3,1,2)) 
        #print('inputs.max() a',inputs.max())
        #print('inputs.shape d',inputs.shape)
        inputs_tensor=torch.tensor(inputs,dtype=torch.float)
        for i in range(inputs_tensor.shape[0]):
            x=transforms.functional.normalize(inputs_tensor[i],mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            #print('transform x shape',x.shape)
            inputs_tensor[i] = x
        #print('inputs_tensor.max() b',inputs_tensor.max().item())
        return inputs_tensor;




