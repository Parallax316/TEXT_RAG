# MongoDB Atlas connection utility for multimodal RAG system

import os
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env at project root
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

MONGODB_URI = os.getenv('MONGODB_ATLAS_URI')
MONGODB_DB = os.getenv('MONGODB_ATLAS_DB', 'rag_multimodal')
EMBEDDINGS_COLLECTION = os.getenv('MONGODB_EMBEDDINGS_COLLECTION', 'embeddings')

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB]
embeddings_col = db[EMBEDDINGS_COLLECTION]

# System config collection for storing current selected collection, etc.
system_config_col = db[os.getenv('MONGODB_SYSTEM_CONFIG_COLLECTION', 'system_config')]

def insert_embedding(document: dict):
    return embeddings_col.insert_one(document)

def find_embeddings(query: dict):
    return list(embeddings_col.find(query))

def update_embedding(query: dict, update: dict):
    return embeddings_col.update_one(query, {'$set': update})

def delete_embedding(query: dict):
    return embeddings_col.delete_one(query)

def list_collections():
    return db.list_collection_names()

def create_collection(name: str):
    db.create_collection(name)

def drop_collection(name: str):
    db.drop_collection(name)

def set_system_config(key: str, value: str):
    system_config_col.update_one({"key": key}, {"$set": {"value": value}}, upsert=True)

def get_system_config(key: str) -> Optional[str]:
    doc = system_config_col.find_one({"key": key})
    return doc["value"] if doc else None
