import multiprocessing
import pandas as pd
import pandas_gbq
import numpy as np
from lightfm import LightFM
from pymongo import MongoClient
from pymongo import UpdateOne
from utils.files_manager import *
from utils.Recommender import RecomManager
from utils.DataManager import DataManager
from utils.Evaluate import evaluate_fm
NUM_THREADS = 16


def read_bq(q):
    return pandas_gbq.read_gbq(q, use_bqstorage_api=True)

def split_df(df_, n_cores=multiprocessing.cpu_count()):
    """Spliting DataFrame into chunks"""

    batch_size = 1 if round(df_.shape[0]/n_cores) < 1 else round(df_.shape[0]/n_cores)
    rest = df_.shape[0]%batch_size 
    cumulative = 0
    for i in range(n_cores):
        cumulative += batch_size
        if i == n_cores-1:
            yield df_.iloc[batch_size*i:cumulative+rest]
        else:
           yield df_.iloc[batch_size*i:cumulative]


def write_mongo(df_,param):
    """Update DataFrame to mongoDB"""

    print('Starting process')
    if param['PASSWORD'] == '':
        client = MongoClient(host=param['HOST'],port=param['PORT'])
                                    
    else:  
        try:
            client = MongoClient(host=param['HOST'],port=param['PORT'],
                                        username=param['USER_NAME'],password=param['PASSWORD'],
                                        authSource=param['AUTHSOURCE'],
                                        authMechanism=param['AUTHMECHANISM'],replicaset=param['REPLICA_SET'])
        except Exception as e:
            print("Connection Faild!:",e)

    db = client[param['DB_NAME']]
    collection = db[param['COLLECTION_NAME']]
    
    bulk_operation = []
    for i in range(df_.shape[0]):
        
        reco_list = [x for x in df_.iloc[i,1:].to_list() if x==x]
        bulk_operation.append(UpdateOne({"_id": int(df_.iloc[i,0])}, {'$set': {'recomendation': reco_list}},upsert=True))
    
    try:
        collection.bulk_write(bulk_operation, ordered=False)  

    except Exception as e:
        print("Write_bulk Failed!", e)          
            
    print('Completed process',df_.shape)
    client.close()


def write_mongo_parallel(_df,param):
    processes = [ multiprocessing.Process(target=write_mongo, 
                                      args=(d,param )) for d in split_df(_df)]

    for process in processes:
        process.start()
    
    for process in processes:
        process.join()
        



def train_fm(train_, epochs=30, items_features=None, users_features=None, loss='warp', n_comp=64):
    model_ = LightFM(loss=loss, no_components=n_comp)
    model_.fit(train_, epochs=epochs,
               item_features=items_features,
               user_features=users_features,
               num_threads=16,
               verbose=True)
    return model_

def main(config_file):
    uid = read_bq(open_file(config_file['USERS_ARTICLES']['query_path']))

    white_list_articles = read_bq(open_file(config_file['WHITE_LIST_ARTICLES']['query_path']))
    #white_list_articles = set(white_list_articles.article_id.tolist())
    
    dm = DataManager(uid,users_col='uid',items_col='article_id')
    dm.fit_df_train(white_list_articles)
    model = train_fm(dm.train)
    rm = RecomManager(model,dm)
    recom = rm.predict()
    evaluate_fm(model, dm.test, dm.train)
    write_mongo_parallel(recom,config_file['MONGO_DB'])


def scheduler(config_file):
    print("{} | app started".format(datetime.now()))
    main(config_file)
    schedule.every().day.at(config_file['SCHEDULE']['time']).do(main,config_file)
    
    while True:
        schedule.run_pending()
        sleep(60)
    


if __name__ == 'main':
    config_file_path = generate_path('CONFIG_FILE.json')
    config_file = read_config(config_file_path )
    #scheduler(config_file)
    main(config_file)













