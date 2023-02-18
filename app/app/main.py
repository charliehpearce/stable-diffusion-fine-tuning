from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute


def version():
    return JSONResponse({"version": 1.0})


production_routes = [APIRoute(path="/", endpoint=version, methods=["GET"])]

app = FastAPI(routes=production_routes)
