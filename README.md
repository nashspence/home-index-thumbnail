# Home Index Thumbnail Module

This repository contains a module for [Home Index](https://github.com/nashspence/home-index) that generates WebP thumbnails and preview images. It exposes the XML‑RPC interface expected by Home Index so the service can enrich file metadata with image thumbnails.

## Quick start

The included `docker-compose.yml` launches Home Index, Meilisearch and this module. After [installing Docker](https://docs.docker.com/get-docker/), run:

```bash
docker compose up
```

Place files under `bind-mounts/files` and browse Meilisearch at <http://localhost:7700> once indexing completes.

## Environment variables

The module reads several variables to control thumbnail generation:

| Variable             | Default | Description                                        |
| -------------------- | ------- | -------------------------------------------------- |
| `NAME`               | thumbnail | Module name returned by `hello()`                 |
| `WEBP_METHOD`        | `6`     | WebP compression method (0–6)                     |
| `WEBP_QUALITY`       | `60`    | Quality parameter for WebP output                 |
| `WEBP_ANIMATION_FPS` | `1`     | Frames per second for animated thumbnails         |
| `WEBP_ANIMATION_FRAMES` | `10`  | Maximum frames extracted from a video             |
| `THUMBNAIL_SIZE`     | `150`   | Size of square thumbnail in pixels                |
| `PREVIEW_SIZE`       | `640`   | Maximum dimension for preview image               |

Adjust these values in `docker-compose.yml` or your environment to customise behaviour.

## Usage

Home Index will call this module automatically when it is listed in the `MODULES` environment variable. The module stores generated `thumbnail.webp` and `preview.webp` files under each document's metadata directory.

## Development

Python dependencies are listed in `requirements.txt`. You can install them locally with:

```bash
pip install -r requirements.txt
```

Run the module directly for debugging:

```bash
python packages/home_index_thumbnail/main.py
```

It will start an XML‑RPC server on port 9000 by default.

