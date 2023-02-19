from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from app.connnectors.redis import redis_connection
from app.connnectors.redis.queues import fine_tune_queue
from app.data_models.request_models import FineTuneStableDiffusionRequest
from app.modules.fine_tune_diffuser import DiffusionPipeline


async def version():
    return JSONResponse({"version": 1.0})


def fine_tune_model(request: FineTuneStableDiffusionRequest):
    """
    Run the fine tuning pieline
    """
    tuning_pipeline = DiffusionPipeline(
        dataset_uuid=request.dataset_uuid, config_uuid=request.config_uuid
    )
    fine_tune_queue.enqueue(tuning_pipeline.run)

    return "OK"


production_routes = [
    APIRoute(path="/", endpoint=version, methods=["GET"]),
    APIRoute(
        path="/api/v1/fine_tune_stable_diffusion",
        endpoint=fine_tune_model,
        methods=["POST"],
    ),
]

app = FastAPI(routes=production_routes)


@app.on_event("shutdown")
async def on_shutdown():
    redis_connection.close()
