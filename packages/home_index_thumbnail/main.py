# region "debugpy"


import os

if str(os.environ.get("DEBUG", "False")) == "True":
    import debugpy

    debugpy.listen(("0.0.0.0", 5678))

    if str(os.environ.get("WAIT_FOR_DEBUGPY_CLIENT", "False")) == "True":
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
        print("Debugger attached.")
        debugpy.breakpoint()


# endregion
# region "import"


import json
import logging
import ffmpeg
import os
import io

from wand.image import Image as WandImage
from PIL import Image as PILImage
from home_index_module import run_server
from webp import WebPPicture, WebPConfig, WebPAnimEncoder, WebPAnimEncoderOptions


# endregion
# region "config"


VERSION = 1
NAME = os.environ.get("NAME", "thumbnail")
WEBP_METHOD = int(os.environ.get("WEBP_METHOD", 6))
WEBP_QUALITY = int(os.environ.get("WEBP_QUALITY", 60))
WEBP_ANIMATION_FPS = int(os.environ.get("WEBP_ANIMATION_FPS", 2))
WEBP_ANIMATION_FRAMES = int(os.environ.get("WEBP_ANIMATION_FRAMES", 10))
THUMBNAIL_SIZE = int(os.environ.get("THUMBNAIL_SIZE", 150))
PREVIEW_SIZE = int(os.environ.get("PREVIEW_SIZE", 150))

# endregion
# region "read images"


def resize_image_bytes_maintain_aspect(img_bytes, max_dimension):
    with WandImage(blob=img_bytes) as img:
        width, height = img.width, img.height
        largest_side = max(width, height)
        if largest_side > max_dimension:
            ratio = max_dimension / float(largest_side)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            img.resize(new_width, new_height)
        return img.make_blob("PNG")


def resize_image_bytes_center_crop_square(img_bytes, size):
    with WandImage(blob=img_bytes) as img:
        width, height = img.width, img.height
        if width > height:
            offset = (width - height) // 2
            img.crop(left=offset, top=0, width=height, height=height)
        else:
            offset = (height - width) // 2
            img.crop(left=0, top=offset, width=width, height=width)
        img.resize(size, size)
        return img.make_blob("PNG")


def image_to_static_webp(img_bytes, out_file, quality, method):
    pil_img = PILImage.open(io.BytesIO(img_bytes))
    pic = WebPPicture.from_pil(pil_img)
    c = WebPConfig.new(quality=quality, method=method)
    wpd = pic.encode(c)
    with open(out_file, "wb") as f:
        f.write(wpd.bytes)


def images_to_animated_webp(
    images_bytes, out_file, width, height, fps, quality, method
):
    e = WebPAnimEncoder.new(width, height, WebPAnimEncoderOptions.new())
    c = WebPConfig.new(quality=quality, method=method)
    frame_duration_ms = int(1000 / fps)
    timestamp = 0
    for frame_bytes in images_bytes:
        pil_img = PILImage.open(io.BytesIO(frame_bytes))
        pic = WebPPicture.from_pil(pil_img)
        e.add(pic, timestamp, c)
        timestamp += frame_duration_ms
    wpd = e.assemble()
    with open(out_file, "wb") as f:
        f.write(wpd.bytes)


def capture_frames(video_path, frame_count):
    probe_info = ffmpeg.probe(video_path)
    video_stream = next(x for x in probe_info["streams"] if x["codec_type"] == "video")
    total_frames = int(video_stream["nb_frames"])
    stride = max(total_frames // frame_count, 1)
    out, _ = (
        ffmpeg.input(video_path)
        .filter_("select", f"not(mod(n,{stride}))")
        .output(
            "pipe:", vcodec="png", format="image2pipe", vframes=frame_count, vsync="vfr"
        )
        .run(capture_stdout=True, capture_stderr=True)
    )
    return out


def split_pngs(png_data):
    frames = []
    png_signature = b"\x89PNG\r\n\x1a\n"
    idx = 0
    data_len = len(png_data)
    while idx < data_len:
        if png_data[idx : idx + 8] == png_signature:
            start = idx
            idx += 8
            while idx < data_len - 8 and png_data[idx : idx + 8] != png_signature:
                idx += 1
            frames.append(png_data[start:idx])
        else:
            idx += 1
    return frames


def create_webp_resize(
    in_path, out_path, frames, fps, quality, size, method, resize_func
):
    try:
        with WandImage(filename=in_path) as im:
            if len(im.sequence) > 1:
                frames_bytes = []
                for frame in im.sequence:
                    with WandImage(image=frame) as fr:
                        fr_bytes = fr.make_blob("PNG")
                        frames_bytes.append(resize_func(fr_bytes, size))
                images_to_animated_webp(
                    frames_bytes, out_path, size, size, fps, quality, method
                )
            else:
                single_png = im.make_blob("PNG")
                resized_png = resize_func(single_png, size)
                image_to_static_webp(resized_png, out_path, quality, method)
        return
    except:
        pass
    try:
        raw_png_data = capture_frames(in_path, frames)
        png_frames = split_pngs(raw_png_data)
        if len(png_frames) > 1:
            frames_bytes = [resize_func(p, size) for p in png_frames]
            images_to_animated_webp(
                frames_bytes, out_path, size, size, fps, quality, method
            )
        elif len(png_frames) == 1:
            resized_png = resize_func(png_frames[0], size)
            image_to_static_webp(resized_png, out_path, quality, method)
    except:
        pass


def create_webp_thumbnail(
    in_path, out_path, thumb_size=150, frames=10, fps=2, quality=60, method=6
):
    create_webp_resize(
        in_path,
        out_path,
        frames,
        fps,
        quality,
        thumb_size,
        method,
        resize_image_bytes_center_crop_square,
    )


def create_webp_preview(
    in_path, out_path, preview_size=640, frames=10, fps=2, quality=60, method=6
):
    create_webp_resize(
        in_path,
        out_path,
        frames,
        fps,
        quality,
        preview_size,
        method,
        resize_image_bytes_maintain_aspect,
    )


# endregion
# region "hello"


def hello():
    return {
        "name": NAME,
        "version": VERSION,
        "filterable_attributes": [f"{NAME}.text"],
        "sortable_attributes": [],
    }


# endregion
# region "check/run"


def check(file_path, document, metadata_dir_path):
    version_path = metadata_dir_path / "version.json"
    version = None
    if version_path.exists():
        with open(version_path, "r") as file:
            version = json.load(file)

    if version and version["version"] == VERSION:
        return False
    try:
        with WandImage(filename=file_path):
            return True
    except Exception:
        try:
            ffmpeg.probe(file_path)
            return True
        except Exception:
            return False


def run(file_path, document, metadata_dir_path):
    global reader
    logging.info(f"start {file_path}")

    version_path = metadata_dir_path / "version.json"
    thumbnail_path = metadata_dir_path / "thumbnail.webp"
    preview_path = metadata_dir_path / "preview.webp"

    exception = None
    try:
        create_webp_thumbnail(
            file_path,
            thumbnail_path,
            thumb_size=THUMBNAIL_SIZE,
            quality=WEBP_QUALITY,
            method=WEBP_METHOD,
            frames=WEBP_ANIMATION_FRAMES,
            fps=WEBP_ANIMATION_FPS,
        )
        create_webp_preview(
            file_path,
            preview_path,
            preview_size=PREVIEW_SIZE,
            quality=WEBP_QUALITY,
            method=WEBP_METHOD,
            frames=WEBP_ANIMATION_FRAMES,
            fps=WEBP_ANIMATION_FPS,
        )
    except Exception as e:
        exception = e
        logging.exception("failed")

    with open(version_path, "w") as file:
        json.dump({"version": VERSION, "exception": str(exception)}, file, indent=4)

    logging.info("done")
    return document


# endregion

if __name__ == "__main__":
    run_server(NAME, hello, check, run)
