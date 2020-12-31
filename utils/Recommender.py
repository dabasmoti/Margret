import multiprocessing
import numpy as np
import pandas as pd


class RecomManager:
    
    def __init__(self,model,dm, k=20, n_threads=16):
        self.model = model
        
        self.users_indexs = list(dm.user_id_map.keys())
        self.users_ids = list(dm.user_id_map.values())
        
        self.items_indexes = list(dm.item_id_map.keys())
        self.items_ids = list(dm.item_id_map.values())
        self.k = k
        self.n_threads = n_threads
        self.dm = dm

    
    def chunks(self, l, n):
        
        """Yield n number of striped chunks from l."""
        for i in range(0, n):
            yield l[i::n]


    def worker_func(self, users):
        self.recomables_articles = list(self.dm.white_articles_map.keys()) 
        for u in users:
            scores = self.model.predict(u, self.recomables_articles)
            self.sdict[self.dm.user_id_map[u]] = [self.dm.white_articles_map[self.recomables_articles[i]]
                                     for i in np.argsort(-scores)[:self.k]]
            
    def predict(self,sort_by=['uid','publish_time'], recom_col='recom',ascending=False):

        manager = multiprocessing.Manager()
        self.sdict = manager.dict()
        processes = [multiprocessing.Process(target=self.worker_func,
                                         args=(d,))
                 for d in self.chunks(range(self.dm._interactions.shape[0]), self.n_threads)]

        for process in processes:
            process.start()
        
        for process in processes:
            process.join()
            
        
        self.recom_df = pd.DataFrame(dict(self.sdict).items(), columns=[self.dm.users_col, recom_col])
        
        
        self.recom_df= self.recom_df.explode(recom_col).reset_index(drop=True)
        
        self.recom_df = self.recom_df.merge(self.dm.white_list.set_index(self.dm.items_col), left_on=recom_col, right_index=True)
        
        self.recom_df = self.recom_df.sort_values(by=sort_by, ascending=ascending)
        
        self.recom_df = self.recom_df.groupby('uid',as_index=False).agg({'recom':list})
        self.recom_df = pd.concat([self.recom_df[self.dm.users_col], \
                                         pd.DataFrame(self.recom_df[recom_col].values.tolist())],axis=1)
        
        self.recom_df.columns = [f'_{str(col)}_' if isinstance(col,int) else col for col in self.recom_df.columns ]
        
        return self.recom_df


