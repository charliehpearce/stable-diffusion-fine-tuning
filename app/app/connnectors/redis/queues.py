from enum import Enum
from rq import Queue
from .instance import redis_connection


class QueueNames(Enum):
    FINE_TUNE_QUEUE = "fine_tune_queue"


fine_tune_queue = Queue(
    name=QueueNames.FINE_TUNE_QUEUE.value,
    connection=redis_connection,
    default_timeout=60 * 60 * 2,
)  # seconds
