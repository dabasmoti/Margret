
from lightfm.data import Dataset
import pandas as pd



class DataManager(Dataset):
    
    def __init__(self,_data,users_col,items_col):
        """ DataManager class 
         _data : Pandas DataFrame   
         users_col : str, the users columns
         items_col : str, the items columns
        """
        self._data = _data
        self.users_col = users_col
        self.items_col = items_col
        super().__init__()
    
    def fit_on_df(self):
        super().fit(users=self._data[self.users_col].unique(), 
                    items=self._data[self.items_col].unique())
    
    def build_interactions(self):
        self._interactions, self._weights =  super(DataManager, self).build_interactions(
                                        (x[self._data.columns.get_loc(self.users_col)], 
                                         x[self._data.columns.get_loc(self.items_col)]) 
                                        for x in self._data.itertuples(index=False))
        
        self.user_id_map = {v: k for k, v in super(DataManager,self).mapping()[0].items()}
        
        self.item_id_map = {v: k for k, v in super(DataManager,self).mapping()[2].items()}
        
    def map_white_list(self,white_list=None):
        self.white_list = white_list
        self.white_list_articles_ids = set(white_list[self.items_col]) if  isinstance(white_list,pd.DataFrame) else self._data[self.items_col]
        self.white_articles_map = {k: v for k,
                      v in self.item_id_map.items() if v in self.white_list_articles_ids}
    
    def split_train_test(self,test_percentage=0.2):
        self.train, self.test = random_train_test_split(self._interactions, test_percentage=test_percentage)
        
    def fit_df_train(self,white_list):
        self.fit_on_df()
        self.build_interactions()
        self.map_white_list(white_list)
        self.split_train_test()
    


















