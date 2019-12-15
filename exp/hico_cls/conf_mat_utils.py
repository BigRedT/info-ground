import os
import numpy as np

import utils.io as io
from global_constants import hico_paths


class ConfMatAggregator():
    def __init__(self,num_classes=600):
        super().__init__()
        self.agg_scores = np.zeros([600,600])
        self.count = np.zeros([600,1])
        self.hoi_list = io.load_json_object(os.path.join(
            hico_paths['proc_dir'],
            hico_paths['hoi_list_json']))

    def update(self,pos_labels,scores):
        scores = np.expand_dims(scores,0)
        self.agg_scores[pos_labels==1] = self.agg_scores[pos_labels==1] + scores
        self.count[pos_labels==1] = self.count[pos_labels==1] + 1
        
    @property
    def conf_mat(self):
        conf_mat = self.agg_scores / (self.count + 1e-6)
        return conf_mat

    def get_interactions_for_object(self,object_name):
        hoi_ids = []
        interactions = []
        for i,hoi in enumerate(self.hoi_list):
            if hoi['object']==object_name:
                hoi_ids.append(i)
                interactions.append(hoi['interaction'])
        
        return hoi_ids, interactions

    def get_objects_for_interaction(self,interaction_name):
        hoi_ids = []
        objects = []
        for i,hoi in enumerate(self.hoi_list):
            if hoi['interaction']==interaction_name:
                hoi_ids.append(i)
                objects.append(hoi['object'])
        
        return hoi_ids, objects

    def conf_mat_object(self,object_name,conf_mat=None):
        hoi_ids, interactions = self.get_interactions_for_object(object_name)
        
        if conf_mat is None:
            conf_mat = self.conf_mat

        conf_mat = conf_mat[hoi_ids][:,hoi_ids]
        
        return conf_mat, interactions

    def conf_mat_interaction(self,interaction_name,conf_mat=None):
        hoi_ids, objects = self.get_objects_for_interaction(interaction_name)

        if conf_mat is None:
            conf_mat = self.conf_mat
            
        conf_mat = conf_mat[hoi_ids][:,hoi_ids]
        
        return conf_mat, objects

    def list_of_objects(self):
        object_names = set()
        for hoi in self.hoi_list:
            object_names.add(hoi['object'])
        
        return sorted(list(object_names))

    def list_of_interactions(self):
        interaction_names = set()
        for hoi in self.hoi_list:
            interaction_names.add(hoi['interaction'])
        
        return sorted(list(interaction_names))



if __name__=='__main__':
    conf_mat_agg = ConfMatAggregator()
    object_hoi_ids, interactions = conf_mat_agg.get_interactions_for_object('dog')
    interaction_hoi_ids, objects = conf_mat_agg.get_objects_for_interaction('walk')
    list_of_interactions = conf_mat_agg.list_of_interactions()
    list_of_objects = conf_mat_agg.list_of_objects()
    import pdb; pdb.set_trace()