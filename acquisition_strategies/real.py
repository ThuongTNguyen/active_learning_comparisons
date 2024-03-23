import numpy as np
import faiss
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
import random


class REAL:
    '''Adapt from REAL's pseudo_err_1_4_1 (https://github.com/withchencheng/ECML_PKDD_23_Real/tree/main/real)
    '''

    def __init__(self, num_clusters=20, additional_transform=None):
        self.num_clusters = num_clusters
        self.additional_transform = additional_transform

    def acquire(self, X_L, y_L, X_U, X_L_trf, X_U_trf, clf, size, **kwargs):
        if self.additional_transform:
            X_U_trf_kmeans = self.additional_transform(self, X_L, y_L, X_U, X_L_trf, X_U_trf, clf, size, kwargs)
        else:
            X_U_trf_kmeans = X_U_trf

        if not isinstance(X_U_trf_kmeans, np.ndarray):  # needed for CountVec output
            X_U_trf_kmeans = X_U_trf_kmeans.toarray()

        unlabeled_feat = X_U_trf_kmeans
        unlabeled_pred = clf.predict_proba(X_U_trf)
        # Use representations of current fine-tuned model *CAL*
        N = unlabeled_pred.shape[0]
        ncentroids = self.num_clusters
        unlabeled_pseudo = np.argmax(unlabeled_pred, axis=-1)
        sample_idx, save_idx = [], []

        I = kmeanspp(ncentroids, unlabeled_feat)
        clu_value = [0] * ncentroids  # cluster value, more error, more valuable
        clu_majlbl = [-1] * ncentroids  # cluster value, more error, more valuable
        dis = np.zeros(N)
        # pass 1: fill clu_value
        for i in range(ncentroids):
            clu_sel = (I == i)  # selector for current cluster
            if np.sum(clu_sel) == 0: continue  # ，faiss
            cnt = Counter()
            for z in unlabeled_pseudo[clu_sel]:
                cnt[z] += 1
            # select minority from cnt
            lbl_freq = list(cnt.items())
            lbl_freq.sort(key=lambda x: x[1])
            clu_pseudo = unlabeled_pseudo[clu_sel]
            majlbl = lbl_freq[-1][0]  # the majority label
            clu_majlbl[i] = majlbl
            majscore = unlabeled_pred[clu_sel][:, majlbl]
            dismaj = 1 - majscore
            dis[clu_sel] = dismaj
            nonmaj_sel = clu_pseudo != majlbl
            # clu_value[i] = np.mean(dismaj[nonmaj_sel]) # set i，
            clu_value[i] = np.sum(dismaj[nonmaj_sel])

        # pass 2: sample proportionanlly to clu_value
        cvsm = np.sum(clu_value)
        if cvsm > 0:
            clu_nsample = [int(i / cvsm * size) for i in clu_value]
        else:
            clu_nsample = [int(i) for i in clu_value]
        nmissing = size - np.sum(clu_nsample)
        highclui = np.argsort(clu_value)[::-1][:nmissing]
        for i in highclui:
            clu_nsample[i] += 1

        for i in range(ncentroids):
            clu_sel = (I == i)  # selector for current cluster
            topk = clu_nsample[i]  # TODO: topk > clu_size, qqp
            if topk <= 0: continue
            majlbl = clu_majlbl[i]
            clu_pseudo = unlabeled_pseudo[clu_sel]
            majscore = unlabeled_pred[clu_sel][:, majlbl]
            nonmaj_sel = clu_pseudo != majlbl
            nonmaj_idx = np.arange(len(clu_pseudo))[nonmaj_sel]  #
            npseudoerr = np.sum(nonmaj_sel)
            if npseudoerr > topk:  # topk
                # random or entropy pick topk from w
                picki = random.sample(nonmaj_idx.tolist(), topk)
            else:
                picki = np.argsort(majscore)[:topk]
            tmp = np.arange(len(I))[clu_sel][picki]
            sample_idx += tmp.tolist()

        dis_rank = np.argsort(dis)[::-1]  # big
        i = 0
        # ，
        labeled = [False] * N
        for i in sample_idx:
            labeled[i] = True
        while len(sample_idx) < size:
            j = dis_rank[i]
            if not labeled[j]:
                sample_idx.append(j)
                labeled[j] = True
            i += 1

        assert len(sample_idx) == size
        for i, lbl in enumerate(labeled):
            if not lbl:
                save_idx.append(i)
        return sample_idx  # , save_idx, entropy, I


def kmeanspp(ncentroids, feat):
    """
    K-means++
    Args:
      ncentroids (int):
      feat: [n, dim]
    """
    dim = feat.shape[-1]
    kmeans = MiniBatchKMeans(n_clusters=ncentroids, random_state=0, n_init=3, max_iter=100)  # default is k-means++
    kmeans.fit(feat)
    index = faiss.IndexFlatL2(dim)
    index.add(kmeans.cluster_centers_)
    D, I = index.search(feat, 1)
    I = I.flatten()  # list of cluster assignment for all unlabeled ins
    return I


