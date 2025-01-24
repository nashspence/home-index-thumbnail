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
import subprocess
import mimetypes

from wand.image import Image as WandImage
from PIL import Image as PILImage
from home_index_module import run_server
from webp import WebPPicture, WebPConfig, WebPAnimEncoder


# endregion
# region "config"


VERSION = 1
NAME = os.environ.get("NAME", "thumbnail")
WEBP_METHOD = int(os.environ.get("WEBP_METHOD", 3))
WEBP_QUALITY = int(os.environ.get("WEBP_QUALITY", 60))
WEBP_ANIMATION_FPS = int(os.environ.get("WEBP_ANIMATION_FPS", 1))
WEBP_ANIMATION_FRAMES = int(os.environ.get("WEBP_ANIMATION_FRAMES", 10))
THUMBNAIL_SIZE = int(os.environ.get("THUMBNAIL_SIZE", 150))
PREVIEW_SIZE = int(os.environ.get("PREVIEW_SIZE", 640))

# endregion
# region "read images"


def resize_image_bytes_maintain_aspect(img_bytes, max_dimension):
    with WandImage(blob=img_bytes) as img:
        img.auto_orient()
        width, height = img.width, img.height
        largest_side = max(width, height)
        new_width = None
        new_height = None
        if largest_side > max_dimension:
            ratio = max_dimension / float(largest_side)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            img.resize(new_width, new_height)
        return img.make_blob("PNG"), new_width, new_height


def resize_image_bytes_center_crop_square(img_bytes, size):
    with WandImage(blob=img_bytes) as img:
        img.auto_orient()
        width, height = img.width, img.height
        if width > height:
            offset = (width - height) // 2
            img.crop(left=offset, top=0, width=height, height=height)
        else:
            offset = (height - width) // 2
            img.crop(left=0, top=offset, width=width, height=width)
        img.resize(size, size)
        return img.make_blob("PNG"), size, size


def image_to_static_webp(img_bytes, out_file, quality, method):
    b, w, h = img_bytes
    pil_img = PILImage.open(io.BytesIO(b))
    pic = WebPPicture.new(w, h).from_pil(
        PILImage.alpha_composite(
            PILImage.new("RGBA", pil_img.size, (255, 255, 255, 255)),
            pil_img.convert("RGBA"),
        )
    )
    c = WebPConfig.new(quality=quality, method=method)
    wpd = pic.encode(c)
    with open(out_file, "wb") as f:
        f.write(wpd.buffer())


def images_to_animated_webp(images_bytes, out_file, fps, quality, method):
    _, w, h = images_bytes[0]
    e = WebPAnimEncoder.new(w, h)
    c = WebPConfig.new(quality=quality, method=method)
    frame_duration_ms = int(1000 / fps)
    timestamp = 0
    for frame_bytes in images_bytes:
        b, _, _ = frame_bytes
        pil_img = PILImage.open(io.BytesIO(b))
        pic = WebPPicture.from_pil(
            PILImage.alpha_composite(
                PILImage.new("RGBA", pil_img.size, (255, 255, 255, 255)),
                pil_img.convert("RGBA"),
            )
        )
        e.encode_frame(pic, timestamp, c)
        timestamp += frame_duration_ms
    wpd = e.assemble(timestamp)
    with open(out_file, "wb") as f:
        f.write(wpd.buffer())


def extract_partitioned_frames(video_path, num_frames):
    probe_info = ffmpeg.probe(video_path)
    duration = float(probe_info["format"]["duration"])
    epsilon = 0.01
    safe_duration = max(0, duration - epsilon)
    timestamps = [i * safe_duration / (num_frames - 1) for i in range(num_frames)]
    frames = []

    def get_frame_at_time(t: float):
        out, _ = (
            ffmpeg.input(video_path, ss=t)
            .output("pipe:", vframes=1, format="image2", vcodec="png")
            .run(capture_stdout=True, capture_stderr=True)
        )
        return out

    for i, t in enumerate(timestamps):
        out = get_frame_at_time(t)
        if i == num_frames - 1 and len(out) == 0:
            fallback_time = t
            step_back = 0.05  # 50ms increments
            retries = 10  # Try stepping back up to 10 times
            while len(out) == 0 and fallback_time > 0 and retries > 0:
                fallback_time -= step_back
                out = get_frame_at_time(fallback_time)
                retries -= 1
        frames.append(out)
    return frames


def extract_keyframes(video_path, num_frames=10):
    command = [
        "ffprobe",
        "-skip_frame",
        "nokey",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=pts_time",
        "-print_format",
        "json",
        video_path,
    ]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    frame_data = json.loads(result.stdout)
    frames = sorted(
        [float(frame["pts_time"]) for frame in frame_data.get("frames", [])]
    )
    keyframe_count = len(frames)
    if keyframe_count == 0:
        raise ValueError("no keyframes in video")
    if keyframe_count <= num_frames:
        num_frames = keyframe_count
        chosen = range(num_frames)
    else:
        step = len(frames) / num_frames
        chosen = [int(i * step) for i in range(num_frames)]
    exprs = [f"eq(n\\,{c})" for c in chosen]
    select_expr = "+".join(exprs)

    cmd = [
        "ffmpeg",
        "-skip_frame",
        "nokey",
        "-i",
        video_path,
        "-vf",
        f"select={select_expr}",
        "-frames:v",
        str(num_frames),
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:",
    ]

    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frames_data = []
    while True:
        signature = pipe.stdout.read(8)
        if not signature:
            break
        if len(signature) < 8 or signature[:4] != b"\x89PNG":
            break
        chunk_data = bytearray(signature)
        while True:
            length_bytes = pipe.stdout.read(4)
            if len(length_bytes) < 4:
                break
            chunk_data.extend(length_bytes)
            length = int.from_bytes(length_bytes, "big")
            chunk_type = pipe.stdout.read(4)
            if len(chunk_type) < 4:
                break
            chunk_data.extend(chunk_type)
            chunk_payload = pipe.stdout.read(length + 4)
            if len(chunk_payload) < length + 4:
                break
            chunk_data.extend(chunk_payload)
            if chunk_type == b"IEND":
                break
        frames_data.append(bytes(chunk_data))
        if len(frames_data) >= num_frames:
            break
    pipe.stdout.close()
    stderr_output = pipe.stderr.read()
    pipe.stderr.close()
    pipe.wait()

    if stderr_output:
        logging.debug(stderr_output.decode().strip())

    return frames_data


def create_webp_resize(
    mimetype, in_path, out_path, frames, fps, quality, size, method, resize_func
):
    if not mimetype.startswith("video/"):
        with WandImage(filename=in_path) as im:
            im.auto_orient()
            if len(im.sequence) > 1:
                frames_bytes = []
                for frame in im.sequence:
                    with WandImage(image=frame) as fr:
                        fr_bytes = fr.make_blob("PNG")
                        frames_bytes.append(resize_func(fr_bytes, size))
                images_to_animated_webp(frames_bytes, out_path, fps, quality, method)
            else:
                single_png = im.make_blob("PNG")
                resized_png = resize_func(single_png, size)
                image_to_static_webp(resized_png, out_path, quality, method)
    else:
        try:
            png_frames = extract_keyframes(in_path, frames)
        except Exception as e:
            png_frames = []
            logging.warning(
                f'failed to extract keyframes because "{str(e)}", trying full-decode frame extraction.'
            )
        if not png_frames:
            png_frames = extract_partitioned_frames(in_path, frames)
        if len(png_frames) > 1:
            frames_bytes = [resize_func(p, size) for p in png_frames]
            images_to_animated_webp(frames_bytes, out_path, fps, quality, method)
        elif len(png_frames) == 1:
            resized_png = resize_func(png_frames[0], size)
            image_to_static_webp(resized_png, out_path, quality, method)


def create_webp_thumbnail(
    mimetype, in_path, out_path, thumb_size, frames, fps, quality, method
):
    create_webp_resize(
        mimetype,
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
    mimetype, in_path, out_path, preview_size, frames, fps, quality, method
):
    create_webp_resize(
        mimetype,
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


def get_supported_formats():
    result = subprocess.run(
        ["identify", "-list", "format"], capture_output=True, text=True
    )
    lines = result.stdout.splitlines()
    supported = set()
    for line in lines:
        line = line.strip()
        if line and not line.startswith("-") and not line.startswith("Format"):
            parts = line.split()
            if len(parts) >= 2:
                ext = parts[0].lower().rstrip("*")
                modes = parts[2]
                if "r" in modes:
                    supported.add(ext)
    return supported


SUPPORTED_FORMATS = get_supported_formats()


def get_extensions_from_mime(mime_type):
    return mimetypes.guess_all_extensions(mime_type)


def can_wand_open(mime_type):
    extensions = get_extensions_from_mime(mime_type)
    for ext in extensions:
        if ext.lstrip(".").lower() in SUPPORTED_FORMATS:
            return True
    return False


def check(file_path, document, metadata_dir_path):
    version_path = metadata_dir_path / "version.json"
    version = None

    if version_path.exists():
        with open(version_path, "r") as file:
            version = json.load(file)

    if version and version.get("version") == VERSION:
        return False

    if document["type"].startswith("audio/") or document["type"].startswith(
        "multipart/appledouble"
    ):
        return False

    return can_wand_open(document["type"])


def run(file_path, document, metadata_dir_path):
    global reader
    logging.info(f"start {file_path}")

    version_path = metadata_dir_path / "version.json"
    thumbnail_path = metadata_dir_path / "thumbnail.webp"
    preview_path = metadata_dir_path / "preview.webp"

    exception = None
    try:
        create_webp_thumbnail(
            document["type"],
            file_path,
            thumbnail_path,
            thumb_size=THUMBNAIL_SIZE,
            quality=WEBP_QUALITY,
            method=WEBP_METHOD,
            frames=WEBP_ANIMATION_FRAMES,
            fps=WEBP_ANIMATION_FPS,
        )
        create_webp_preview(
            document["type"],
            file_path,
            preview_path,
            preview_size=PREVIEW_SIZE,
            quality=WEBP_QUALITY,
            method=WEBP_METHOD,
            frames=WEBP_ANIMATION_FRAMES,
            fps=WEBP_ANIMATION_FPS,
        )
    except FileNotFoundError as e:
        raise e
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
