"""Small HTTP API around the saved recommender.

    uvicorn api.main:app --reload      ->  http://localhost:8000/docs
"""
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.config import AGE_GROUPS, K
from src.pipeline import Recommender

app = FastAPI(title="Movie recommender (MovieLens 1M)", version="1.0",
              description="Hybrid item-kNN + ALS + popularity, with cold-start routing. "
                          "Model state: ratings up to 2 Dec 2000.")
Model = Literal["hybrid", "item_knn", "als", "content", "popularity"]


@lru_cache(maxsize=1)
def rec() -> Recommender:
    return Recommender.load()


class Recommendation(BaseModel):
    item_id: int
    title: str
    year: int
    genres: list[str]
    score: float
    reason: str = Field(description="Human-readable explanation")
    because: list[int] = Field(description="Liked movie ids cited in the explanation")
    model: str = Field(description="Route that produced the list (cold-start users are re-routed)")
    drivers: dict[str, float] | None = Field(None, description="Hybrid only: share of the score from "
                                             "your movies (item-kNN), taste vectors (ALS), popularity, genres")


class NewUser(BaseModel):
    liked_movie_ids: list[int] = Field(default_factory=list, max_length=20, examples=[[2571, 1196, 260]])
    gender: Literal["M", "F"] | None = None
    age: int | None = Field(None, description=f"MovieLens age group code: {list(AGE_GROUPS)}")
    k: int = Field(K, ge=1, le=50)


@app.get("/health")
def health():
    r = rec()
    return {"status": "ok", "users": len(r.known_users), "movies": r.cat.n_items, "params": r.params}


@app.get("/recommend/{user_id}", response_model=list[Recommendation])
def recommend_user(user_id: int, k: int = Query(K, ge=1, le=50), model: Model = "hybrid"):
    r = rec()
    if user_id not in r.uidx:
        raise HTTPException(404, f"user {user_id} has no ratings before the cutoff; "
                                 "use POST /recommend/new-user")
    return r.recommend(user_id=user_id, k=k, model=model)


@app.post("/recommend/new-user", response_model=list[Recommendation])
def recommend_new_user(body: NewUser):
    r = rec()
    unknown = [i for i in body.liked_movie_ids if i not in r.cat.item_index]
    if unknown:
        raise HTTPException(422, f"unknown movie ids: {unknown}")
    if body.age is not None and body.age not in AGE_GROUPS:
        raise HTTPException(422, f"age must be one of {list(AGE_GROUPS)}")
    return r.recommend(liked_items=body.liked_movie_ids, gender=body.gender, age=body.age, k=body.k)


@app.get("/similar/{movie_id}")
def similar(movie_id: int, k: int = Query(K, ge=1, le=50)):
    r = rec()
    if movie_id not in r.cat.item_index:
        raise HTTPException(404, f"movie {movie_id} not in catalogue")
    m = r.cat.movies.iloc[r.cat.item_index[movie_id]]
    return {"movie": {"item_id": movie_id, "title": m["clean_title"], "year": int(m["year"])},
            "similar": r.similar(movie_id, k)}
