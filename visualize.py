import json
import matplotlib.pyplot as plt
import sys
import os
from collections import OrderedDict
import matplotlib


class visualize_mmdetection():
    def __init__(self, path):
        self.log = open(path)
        self.dict_list = list()
        self.loss_bbox = list()
        self.loss_cls = list()
        self.loss = list()


    def load_data(self):
        for line in self.log:
            info = json.loads(line)
            self.dict_list.append(info)
        #print(self.dict_list[-1]['mode'])
        for i in range(1, len(self.dict_list)):
            mode = dict(self.dict_list[i]).get('mode')
            if mode == 'train':
                #print(dict(self.dict_list[i]))
            #for key, value in dict(self.dict_list[i]).items():
                # ------------find key for every iter-------------------#



                loss_bbox_value = dict(self.dict_list[i])['loss_bbox']
                loss_cls_value = dict(self.dict_list[i])['loss_cls']
                loss_value = dict(self.dict_list[i])['loss']
                # -------------list append------------------------------#
                self.loss_bbox.append(loss_bbox_value)
                self.loss_cls.append(loss_cls_value)
                self.loss.append(loss_value)

                # -------------clear repeated value---------------------#
        self.loss_bbox = list(OrderedDict.fromkeys(self.loss_bbox))
        self.loss_cls = list(OrderedDict.fromkeys(self.loss_cls))
        self.loss = list(OrderedDict.fromkeys(self.loss))


    def show_chart(self):
        plt.rcParams.update({'font.size': 15})

        plt.figure(figsize=(20, 20))  

        plt.subplot(221, title='loss_cls', ylabel='loss')
        plt.plot(self.loss_cls)
        plt.subplot(222, title='loss_bbox', ylabel='loss')
        plt.plot(self.loss_bbox)
        plt.subplot(223, title='total loss', ylabel='loss')
        plt.plot(self.loss)

        plt.show()
  
        #plt.savefig('outputs/results/result.png')


if __name__ == '__main__':
    x = visualize_mmdetection('/home/chen/OD/mmdet_tutorial/tutorial_exps/tood_licence/None.log.json')
    x.load_data()
    x.show_chart()


