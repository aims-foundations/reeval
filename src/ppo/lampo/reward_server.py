"""
This is reward model server main process.
"""

import argparse
from contextlib import asynccontextmanager
from typing import List

import uvicorn
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import reward_model, utils

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--config", type=str, default="")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=4324)
args = parser.parse_args()


@asynccontextmanager
async def lifespan(_: "FastAPI"):
    """
    Collects GPU memory
    """
    yield
    utils.torch_gc()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

reward_model = reward_model.get_reward_model(model_name=args.model, config=args.config)


class ScoreResponse(BaseModel):
    """
    Response data class
    """

    scores: List[float] = []


class ComputeRequest(BaseModel):
    """
    Request data class
    """

    model: str
    messages: List[List[str]]


@app.post(
    "/",
    response_model=ScoreResponse,
    status_code=status.HTTP_200_OK,
)
async def compute_reward(request: ComputeRequest):
    """
    This function is used to call reward model for computing reward scores.
    """
    return ScoreResponse(scores=await reward_model.compute(request.messages))


def run_server():
    """
    Main function for starting reward server
    """
    print(f"Visit http://localhost:{args.port}/docs for API document.")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    run_server()
