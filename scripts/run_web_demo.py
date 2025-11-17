#!/usr/bin/env python3
"""Script pour lancer le serveur web FastAPI."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("polytopia_jax.web.api:app", host="0.0.0.0", port=8000, reload=True)

