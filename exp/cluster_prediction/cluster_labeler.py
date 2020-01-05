from sklearn.cluster import MiniBatchKMeans


class ClusterLabeler():
    def __init__(
            self,
            cluster_centers,
            labels,
            active_nouns):
        self.kmeans = MiniBatchKMeans(
            n_clusters=cluster_centers.shape[0],
            random_state=0,
            batch_size=1)
        self.kmeans.cluster_centers_ = cluster_centers
        self.labels = labels
        self.label_to_idx = {l:i for i,l in enumerate(self.labels)}
        self.active_nouns = active_nouns
        self.all_active_words = set()
        for words in self.active_nouns:
            self.all_active_words.update(set(words))

    def cluster_id_to_label(self,word,cluster_id):
        if word in self.active_nouns[cluster_id]:
            label = f'{word}_{cluster_id}'
        elif word in self.all_active_words:
            label = f'{word}_-1'
        else:
            if len(self.active_nouns[cluster_id]) > 0:
                active_noun = self.active_nouns[cluster_id][0]
                label = f'{active_noun}_{cluster_id}'
            else:
                label = None

        return label

    def get_cluster_id(self,feat):
        # feat is expected to be (768,)
        assert(len(feat.shape)==1),'feat must have shape (768,)'
        return self.kmeans.predict(feat.reshape(1,-1))[0]

    def get_label(self,word,feat):
        cluster_id = self.get_cluster_id(feat)
        label = self.cluster_id_to_label(word,cluster_id)
        return label

    def get_idx(self,label):
        if label in self.label_to_idx:
            return self.label_to_idx[label]
        else:
            return -1
    