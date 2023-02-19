from redis import Redis
import os

REDIS_HOST = os.environ["REDIS_HOST"]
REDIS_PORT = int(os.environ["REDIS_PORT"])
REDIS_DB = int(os.environ["REDIS_DB"])

redis_connection = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
