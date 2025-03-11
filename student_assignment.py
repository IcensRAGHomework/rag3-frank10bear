import datetime
import chromadb
import traceback
import pandas
import os

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"
csv_file_name = "COA_OpenData.csv"
def generate_hw01():
    print("run generate_hw01")
    #persistentClient = PersistentClient("./chroma", )
    
    if not os.path.exists(csv_file_name):
        print(f"File {csv_file_name} does not exist.")
        return

    collection = getOrCreateCollection("")
    if collection.count() == 0:
        print("collection is empty")
        dataframe = pandas.read_csv(csv_file_name)
        print("idx=")
        for idx, row in dataframe.iterrows():
            print(str(idx) + ", ")
            metadata = {
                "file_name": csv_file_name,
                "name": row["Name"],
                "type": row["Type"],
                "address": row["Address"],
                "tel": row["Tel"],
                "city": row["City"],
                "town": row["Town"],
                "date": int(datetime.datetime.strptime(row['CreateDate'], '%Y-%m-%d').timestamp())
            }
            collection.add(
                ids=[str(idx)],
                metadatas=[metadata],
                documents=[row["HostWords"]]
            )
        print("insert " + str(collection.count) + " count(s) to collection. finish")
        return collection
    else:
        print("collection is not empty, skip insert, then return " + str(collection.count()) + " count(s)")
        return collection
    
    
def generate_hw02(question, city, store_type, start_date, end_date):
    collection = generate_hw01()
    query_results = collection.query(
        query_texts=[question],
        n_results=10,
        include=["metadatas", "distances"],
        where={
            "$and": [
                {"date": {"$gte": int(start_date.timestamp())}}, # greater than or equal
                {"date": {"$lte": int(end_date.timestamp())}}, # less than or equal
                {"type": {"$in": store_type}},
                {"city": {"$in": city}}
            ]
        }
        )
    print(len((query_results['ids'])[0]))
    print(query_results.keys())
    filtered_similarity_store_name = []
    for index in range(len(query_results['ids'])):
        for metadata, distance in zip(query_results['metadatas'][index], query_results['distances'][index]):
            similarity = 1 - distance
            print("metadata name="+metadata['name']+", distance="+str(distance) + ", similarity=" + str(similarity))
            if similarity > 0.8:
                filtered_similarity_store_name.append([metadata['name'], similarity])
    print(filtered_similarity_store_name)
    filtered_similarity_store_name.sort(key=lambda x: x[1], reverse=True)
    filter_store_name, filter_similarity = zip(*filtered_similarity_store_name)
    print(filter_store_name)
    return filter_store_name
       
def generate_hw03(question, store_name, new_store_name, city, store_type):
    pass
    
def getOrCreateCollection(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection

if __name__ == "__main__" :
    #generate_hw01()
    generate_hw02("我想要找有關茶餐點的店家", ["宜蘭縣", "新北市"], ["美食"], datetime.datetime(2024, 4, 1), datetime.datetime(2024, 5, 1))
    #generate_hw03("我想要找南投縣的田媽媽餐廳，招牌是蕎麥麵", "耄饕客棧", "田媽媽（耄饕客棧）", ["南投縣"], ["美食"])