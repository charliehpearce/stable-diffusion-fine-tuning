from redis.asyncio import Redis
import os

REDIS_HOST = os.environ["REDIS_HOST"]
REDIS_PORT = int(os.environ["REDIS_PORT"])


class RedisInstance:
    def __int__(self):
        self.client: Redis

    def open_connection(self):
        self.client = Redis(host=REDIS_HOST, port=REDIS_PORT)

    async def close_connection(self):
        await self.client.close()


redis_connection = RedisInstance()
