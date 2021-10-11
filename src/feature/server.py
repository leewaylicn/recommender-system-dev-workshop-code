import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import os
import logging
import json
import grpc
import time
from threading import Thread
import sys
from pydantic import BaseModel

app = FastAPI()

# Mandatory variables in envirnment
MANDATORY_ENV_VARS = {
    'REDIS_HOST': 'localhost',
    'REDIS_PORT': 6379,
    'FILTER_PORT': 5200
    }

# Notice channel
rank_notice_to_filter='rank_notice_to_filter'
sleep_interval = 10 #second
pickle_type = 'inverted-list'


class Item(BaseModel):
    id: str
    content: str

def xasync(f):
    def wrapper(*args, **kwargs):
        thr = Thread(target = f, args = args, kwargs = kwargs)
        thr.start()
    return wrapper

# @app.get('/feature/status', tags=["monitoring"])
# def status():
#     logging.info('Collecting status information from server & plugin...')
#     channel = grpc.insecure_channel('localhost:50051')
#     stub = service_pb2_grpc.FilterStub(channel)
#     response = stub.Status(service_pb2.google_dot_protobuf_dot_empty__pb2.Empty())
#
#     statusAny = any_pb2.Any()
#     response.status.Unpack(statusAny)
#
#     pStatus = json.loads(statusAny.value.decode('utf-8'))
#     return {
#         'env': MANDATORY_ENV_VARS,
#         'redis': rCache.connection_status(),
#         'plugin_status': pStatus
#     }

@app.get('/ping', tags=["monitoring"])
def ping(): 
    logging.info('Processing default request...')
    return {'result': 'ping'}


# 用户新发布帖子时后台将通过这个接口post请求，将帖子文本传给feature服务，将帖子文本缓存起来
# feature plugin读取缓存，然后得到帖子价值特征，使得用户新发布的帖子能够进入召回候选集中
# @app.post("/get_post")
# async def get_post(id: str, content: str):
#     # id: 工单id
#     start_time = time.time()
#     info_str = 'query id: {},'.format(id)
#
#     logging.info(info_str)
#
#     print(id, content)
#
#     if write_in_cache(id, content):
#         return {'status': 0}
#     else:
#         logging.warning(f'write {id} into cache failed')
#         return {'status': 1}


@app.post("/get_post")
async def get_post(item: Item):
    # id: 工单id
    start_time = time.time()
    id = item.id
    content = item.content
    info_str = 'query id: {}, '.format(id)

    status = 0
    if not write_in_cache(id, content):
        status = 1
        logging.warning(f'write {id} into cache failed')
    time_spent = (time.time() - start_time) * 1000
    info_str += f'cost time {time_spent:.3f}ms'
    logging.info(info_str)
    return {'status': status}


def write_in_cache(id, content):
    pass


if __name__ == "__main__":

    print('server start')
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    uvicorn.run(app, host="0.0.0.0", port=MANDATORY_ENV_VARS['FILTER_PORT'])
    
   
