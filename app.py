import multiprocessing
import pandas as pd
import pandas_gbq
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from random_split import random_train_test_split
from lightfm.evaluation import precision_at_k, auc_score
NUM_THREADS = 16


def read_bq(q):
    return pandas_gbq.read_gbq(q, use_bqstorage_api=True)


def train_fm(train_, epochs=30, items_features=None, users_features=None, loss='warp', n_comp=64):
    model_ = LightFM(loss=loss, no_components=n_comp)
    model_.fit(train_, epochs=epochs,
               item_features=items_features,
               user_features=users_features,
               num_threads=16,
               verbose=True)
    return model_


def evaluate_fm(model_, te_, tr_,
                items_features=None,
                users_features=None):
    if not tr_.multiply(te_).nnz == 0:
        print('train test interaction are not fully disjoin')

    # Compute and print the AUC score
    train_auc = auc_score(model_, tr_,
                          item_features=items_features,
                          user_features=users_features,
                          num_threads=NUM_THREADS).mean()
    print('Collaborative filtering train AUC: %s' % train_auc)

    test_auc = auc_score(model_, te_,
                         train_interactions=tr_,
                         item_features=items_features,
                         user_features=users_features,
                         num_threads=NUM_THREADS).mean()
    print('Collaborative filtering test AUC: %s' % test_auc)
    p_at_k_train = precision_at_k(model_, tr_,
                                  item_features=items_features,
                                  user_features=users_features,
                                  k=5, num_threads=NUM_THREADS).mean()
    p_at_k_test = precision_at_k(model_, te_, train_interactions=tr_,
                                 item_features=items_features,
                                 user_features=users_features,
                                 k=5, num_threads=NUM_THREADS).mean()

    print("Train precision: %.2f" % p_at_k_train)
    print("Test precision: %.2f" % p_at_k_test)


def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]


def worker_func(model, users, item_ids, uid_map, articles_map_, user_dict, k=20):
    # Do work. This function will have read-only access to
    # the data array.

    # for u in users:
    #     scores = model.predict(np.full(len(item_ids),u),item_ids)
    #     user_dict[u] = [articles_map_[i] for i in  np.argsort(-scores)[:k]]
    for u in users:
        scores = model.predict(u, item_ids)
        user_dict[uid_map[u]] = [articles_map_[item_ids[i]]
                                 for i in np.argsort(-scores)[:k]]


uid_q = " select uid,article_id from `htz-common.Recommendations.fm_train_by_uid`"
uid = read_bq(uid_q)

dataset = Dataset()


dataset.fit(users=uid.uid.unique(),
            items=uid.article_id.unique())

(interactions, weights) = dataset.build_interactions(
    (x.uid, x.article_id) for x in uid.itertuples())
n_users, n_items = interactions.shape

user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()
articles_map = {v: k for k, v in item_id_map.items()}
uids_map = {v: k for k, v in user_id_map.items()}
white_list_articles = read_bq(
    """select article_id,publish_time from  `htz-common.Recommendations.white_list`""")
white_list_articles = set(white_list_articles.article_id.tolist())

white_articles_map = {k: v for k,
                      v in articles_map.items() if v in white_list_articles}

tr, te = random_train_test_split(interactions, test_percentage=0.2)

#model_with_features = train_fm(tr,users_features=user_features,items_features=item_features,loss='warp')
model = train_fm(tr, loss='warp')
evaluate_fm(model, te, tr)

items_ids = list(white_articles_map.keys())

workers = 16
manager = multiprocessing.Manager()
sdict = manager.dict()
processes = [multiprocessing.Process(target=worker_func,
                                     args=(model, d, items_ids, uids_map, white_articles_map, sdict))
             for d in chunks(range(n_users), workers)]

for process in processes:
    process.start()

for process in processes:
    process.join()

recom = dict(sdict)
recom.keys()

recom1 = {uids_map[k]: [i for i in v if i in white_list_articles]
          for k, v in recom.items()}

recom[49509]
uid[uid.uid == 1001].article_id

[i for i in recom[49509] if i in uid[uid.uid == 49509].article_id]

[d for d in chunks(uid.uid.unique(), workers)]
'1.9196961' in white_list_articles

df = pd.DataFrame(recom.items(), columns=['uid', 'recom'])
df = df.explode('recom').reset_index(drop=True)
df = df.merge(white_list_articles.set_index(
    'article_id'), left_on='recom', right_index=True)
df = df.sort_values(by=['uid', 'publish_time'], ascending=False)

df[df.uid == 1001]
g = df.groupby(['uid'])['recom'].unique()
g
