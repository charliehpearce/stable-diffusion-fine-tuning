from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from app.connnectors.redis import redis_connection


async def version():
    return JSONResponse({"version": 1.0})


production_routes = [APIRoute(path="/", endpoint=version, methods=["GET"])]

app = FastAPI(routes=production_routes)


@app.on_event("startup")
def on_startup():
    redis_connection.open_connection()


@app.on_event("shutdown")
async def on_shutdown():
    await redis_connection.close_connection()
