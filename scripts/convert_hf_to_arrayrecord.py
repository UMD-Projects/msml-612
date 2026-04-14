"""
Convert HuggingFace datasets to ArrayRecord format for FlaxDiff/Grain training pipeline.

Supports: DiffusionDB, CommonCatalog, JourneyDB, or any HF dataset with image+text columns.

Downloads data in batches, converts images to JPEG, packs into ArrayRecord shards
using the same format as img2dataset (pack_dict_of_byte_arrays), uploads to GCS,
and cleans up local files. Max ~5GB on disk at any time.

Uses existing FlaxDiff data-processing.py utilities (pack_dict_of_byte_arrays,
ArrayRecordSampleWriter pattern).

Usage:
    # DiffusionDB (downloads zips in batches from HF)
    HF_TOKEN=xxx python convert_hf_to_arrayrecord.py \\
        --dataset diffusiondb --subset 2m_all \\
        --output_folder gs://bucket/arrayrecord2/diffusiondb

    # Any HF dataset with image + text columns
    HF_TOKEN=xxx python convert_hf_to_arrayrecord.py \\
        --dataset common-canvas/commoncatalog-cc-by \\
        --image_col jpg --text_col caption \\
        --output_folder gs://bucket/arrayrecord2/commoncatalog
"""

import struct, os, json, io, time, zipfile, argparse
import cv2, numpy as np
from array_record.python.array_record_module import ArrayRecordWriter
import pyarrow, pyarrow.fs


# ---- Reuse pack format from data-processing.py / img2dataset ----

def pack_dict_of_byte_arrays(data_dict):
    packed_data = bytearray()
    for key, byte_array in data_dict.items():
        if not isinstance(key, str):
            raise ValueError("Keys must be strings")
        key_bytes = key.encode('utf-8')
        packed_data.extend(struct.pack('I', len(key_bytes)))
        packed_data.extend(key_bytes)
        packed_data.extend(struct.pack('I', len(byte_array)))
        packed_data.extend(byte_array)
    return bytes(packed_data)


def image_to_jpeg(image_bytes_or_pil, quality=95, target_size=256, max_aspect_ratio=2.4, min_size=100):
    """Convert any image (PIL or bytes) to JPEG bytes via cv2. Resizes to target_size preserving aspect ratio."""
    if hasattr(image_bytes_or_pil, 'save'):
        arr = np.array(image_bytes_or_pil)
    elif isinstance(image_bytes_or_pil, bytes):
        arr = cv2.imdecode(np.frombuffer(image_bytes_or_pil, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        return None
    if arr is None:
        return None
    if len(arr.shape) == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    elif arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    h, w = arr.shape[:2]
    if min(h, w) < min_size:
        return None
    if max(h, w) / min(h, w) > max_aspect_ratio:
        return None

    # Resize: scale so smaller dim == target_size, then center crop
    if h < w:
        new_h = target_size
        new_w = int(w * target_size / h)
    else:
        new_w = target_size
        new_h = int(h * target_size / w)
    arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    y0 = (new_h - target_size) // 2
    x0 = (new_w - target_size) // 2
    arr = arr[y0:y0+target_size, x0:x0+target_size]

    _, encoded = cv2.imencode('.jpg', arr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return encoded.tobytes()


def flush_shard(shard_id, records, output_folder, tmp_dir, oom_shard_count=5):
    """Write records to ArrayRecord shard, upload to GCS if needed, delete temp."""
    os.makedirs(tmp_dir, exist_ok=True)
    shard_name = f"{shard_id:0{oom_shard_count}d}"
    output_file = f"{output_folder}/{shard_name}.array_record"
    if "gs:" in output_folder:
        tmp_file = f"{tmp_dir}/{shard_name}.array_record"
    else:
        tmp_file = output_file

    writer = ArrayRecordWriter(tmp_file, options="group_size:1")
    for r in records:
        writer.write(r)
    writer.close()

    fsize = os.path.getsize(tmp_file)

    if tmp_file != output_file:
        pyarrow.fs.copy_files(tmp_file, output_file, chunk_size=2**24)
        os.remove(tmp_file)

    return fsize


# ---- DiffusionDB: batched zip download ----

def convert_diffusiondb(args):
    """DiffusionDB stores images as zip files at images/part-XXXXXX.zip.

    Each zip contains ~1000 PNG images + one JSON file with prompts/metadata.
    Downloads one zip at a time, extracts, packs into ArrayRecord shards, uploads to GCS.
    """
    from huggingface_hub import hf_hub_download
    import shutil
    from concurrent.futures import ThreadPoolExecutor, as_completed

    REPO = "poloclub/diffusiondb"
    NUM_PARTS = 2000  # 2M images across 2000 zip parts (part-000001 through part-002000)

    shard_id, records, total_written, total_skipped, total_bytes = 0, [], 0, 0, 0
    start_time = time.time()
    num_workers = getattr(args, 'workers', 32)

    for part_id in range(1, NUM_PARTS + 1):
        part_name = f"part-{part_id:06d}"
        zip_filename = f"images/{part_name}.zip"

        # Download zip with automatic retry
        try:
            zip_path = hf_hub_download(
                repo_id=REPO, repo_type="dataset",
                filename=zip_filename, cache_dir=args.tmp_dir,
                force_download=False,
            )
        except Exception as e:
            print(f"[{part_id}/{NUM_PARTS}] SKIP {part_name}: download error: {str(e)[:80]}", flush=True)
            continue

        # Extract zip contents
        try:
            zf = zipfile.ZipFile(zip_path, 'r')
        except Exception as e:
            print(f"[{part_id}/{NUM_PARTS}] SKIP {part_name}: zip error: {str(e)[:80]}", flush=True)
            continue

        # Find the JSON metadata file inside the zip
        names = zf.namelist()
        json_names = [n for n in names if n.endswith('.json')]
        metadata = {}
        if json_names:
            try:
                with zf.open(json_names[0]) as f:
                    metadata = json.load(f)
            except Exception:
                metadata = {}

        # Collect image entries for parallel processing
        img_names = [n for n in names if n.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

        def process_one(img_name):
            try:
                img_bytes = zf.read(img_name)
                jpg_bytes = image_to_jpeg(img_bytes, quality=args.jpeg_quality)
                if jpg_bytes is None:
                    return None
                meta_entry = metadata.get(img_name, {})
                caption = str(meta_entry.get("p", ""))
                meta = {k: meta_entry[k] for k in ["se", "c", "st", "sa"] if k in meta_entry}
                return pack_dict_of_byte_arrays({
                    "key": f"diffdb_{part_id}_{img_name}".encode("utf-8"),
                    "jpg": jpg_bytes,
                    "txt": caption.encode("utf-8"),
                    "meta": json.dumps(meta).encode("utf-8"),
                })
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            results = list(ex.map(process_one, img_names))

        for r in results:
            if r is not None:
                records.append(r)
                total_written += 1
            else:
                total_skipped += 1

        zf.close()

        # Flush shards as they fill
        while len(records) >= args.samples_per_shard:
            fsize = flush_shard(shard_id, records[:args.samples_per_shard],
                                args.output_folder, args.tmp_dir)
            records = records[args.samples_per_shard:]
            total_bytes += fsize
            elapsed = time.time() - start_time
            print(f"  Shard {shard_id}: {fsize/1e6:.0f}MB -> GCS "
                  f"({total_written:,} total, {total_written/elapsed:.1f}/s)", flush=True)
            shard_id += 1

        # Delete local zip after processing
        try:
            os.remove(zip_path)
        except Exception:
            pass

        elapsed = time.time() - start_time
        rate = total_written / max(elapsed, 1)
        print(f"[{part_id}/{NUM_PARTS}] {part_name}: {len(img_names)} imgs processed, "
              f"total_written={total_written:,}, rate={rate:.1f}/s", flush=True)

        # Clean HF cache blobs periodically
        if part_id % 20 == 0:
            cache_blobs = os.path.join(args.tmp_dir, "datasets--poloclub--diffusiondb", "blobs")
            if os.path.exists(cache_blobs):
                shutil.rmtree(cache_blobs, ignore_errors=True)

    return shard_id, records, total_written, total_skipped, total_bytes


# ---- Generic HF dataset: streaming ----

def _process_sample(sample, image_col, text_col, jpeg_quality, dataset_name, idx):
    """Process a single sample — designed for use with multiprocessing."""
    try:
        img = sample.get(image_col)
        if img is None:
            return None
        jpg_bytes = image_to_jpeg(img, quality=jpeg_quality)
        if jpg_bytes is None:
            return None
        caption = str(sample.get(text_col, ""))
        meta = {}
        for k in ["seed", "step", "cfg", "sampler", "width", "height", "key", "sha256"]:
            v = sample.get(k)
            if v is not None and isinstance(v, (int, float, str, bool)):
                meta[k] = v
        return pack_dict_of_byte_arrays({
            "key": f"{dataset_name}_{idx}".encode("utf-8"),
            "jpg": jpg_bytes,
            "txt": caption.encode("utf-8"),
            "meta": json.dumps(meta).encode("utf-8"),
        })
    except Exception:
        return None


def convert_hf_batch_download(args):
    """Download parquet files from HF one at a time, process locally, upload shards.

    Much more reliable than streaming — each file is an independent download with retries.
    No persistent connection needed.
    """
    from huggingface_hub import HfApi, hf_hub_download
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import pandas as pd
    from PIL import Image

    num_workers = getattr(args, 'workers', 32)
    api = HfApi()
    dataset_name = args.dataset.split('/')[-1]

    # List all parquet files in the repo
    print("Listing parquet files in repo...", flush=True)
    all_files = []
    try:
        for item in api.list_repo_tree(args.dataset, repo_type="dataset",
                                        path_in_repo="", recursive=True):
            if hasattr(item, 'rfilename') and item.rfilename.endswith('.parquet'):
                all_files.append(item)
    except Exception as e:
        print(f"Error listing repo: {e}", flush=True)
        # Fallback: try loading with datasets library
        print("Falling back to streaming mode...", flush=True)
        return convert_hf_streaming(args)

    print(f"Found {len(all_files)} parquet files, "
          f"total {sum(f.size for f in all_files)/1e9:.1f}GB", flush=True)

    shard_id, records, total_written, total_skipped, total_bytes = 0, [], 0, 0, 0
    start_time = time.time()
    global_idx = 0

    for fi, pq_file in enumerate(all_files):
        fname = pq_file.rfilename
        fsize_mb = pq_file.size / 1e6

        # Download parquet file locally
        print(f"[{fi+1}/{len(all_files)}] Downloading {fname} ({fsize_mb:.0f}MB)...",
              flush=True, end=" ")
        try:
            local_path = hf_hub_download(
                args.dataset, fname, repo_type="dataset",
                cache_dir=args.tmp_dir, force_download=False,
            )
        except Exception as e:
            print(f"SKIP (download error: {e})", flush=True)
            continue

        # Read parquet and process images
        try:
            df = pd.read_parquet(local_path)
        except Exception as e:
            print(f"SKIP (read error: {e})", flush=True)
            continue

        print(f"{len(df)} rows. Processing...", flush=True)
        batch_samples = []
        for _, row in df.iterrows():
            sample = row.to_dict()
            # Handle image column — could be bytes, dict with 'bytes', or path
            img_data = sample.get(args.image_col)
            if isinstance(img_data, dict) and 'bytes' in img_data:
                sample[args.image_col] = img_data['bytes']
            elif isinstance(img_data, dict) and 'path' in img_data:
                continue  # Can't resolve path references
            batch_samples.append((sample, global_idx))
            global_idx += 1

        # Process batch with thread pool
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _process_sample, s, args.image_col, args.text_col,
                    args.jpeg_quality, dataset_name, idx
                ) for s, idx in batch_samples
            ]
            for f in as_completed(futures):
                r = f.result()
                if r is not None:
                    records.append(r)
                    total_written += 1
                else:
                    total_skipped += 1

        # Flush completed shards
        while len(records) >= args.samples_per_shard:
            fsize = flush_shard(shard_id, records[:args.samples_per_shard],
                                args.output_folder, args.tmp_dir)
            records = records[args.samples_per_shard:]
            total_bytes += fsize
            elapsed = time.time() - start_time
            print(f"  Shard {shard_id}: {fsize/1e6:.0f}MB -> GCS "
                  f"({total_written:,} total, {total_written/elapsed:.1f}/s)", flush=True)
            shard_id += 1

        elapsed = time.time() - start_time
        rate = total_written / max(elapsed, 1)
        print(f"  Progress: {total_written:,} written, {total_skipped:,} skipped, "
              f"{rate:.1f}/s", flush=True)

    return shard_id, records, total_written, total_skipped, total_bytes


def convert_hf_streaming(args):
    """Fallback: stream from HF datasets library."""
    from datasets import load_dataset
    from concurrent.futures import ThreadPoolExecutor, as_completed

    num_workers = getattr(args, 'workers', 32)
    ds = load_dataset(args.dataset, args.subset, split=args.split,
                      streaming=True, trust_remote_code=True)

    shard_id, records, total_written, total_skipped, total_bytes = 0, [], 0, 0, 0
    start_time = time.time()
    dataset_name = args.dataset.split('/')[-1]

    for i, sample in enumerate(ds):
        r = _process_sample(sample, args.image_col, args.text_col,
                            args.jpeg_quality, dataset_name, i)
        if r is not None:
            records.append(r)
            total_written += 1
        else:
            total_skipped += 1

        if len(records) >= args.samples_per_shard:
            fsize = flush_shard(shard_id, records[:args.samples_per_shard],
                                args.output_folder, args.tmp_dir)
            records = records[args.samples_per_shard:]
            total_bytes += fsize
            elapsed = time.time() - start_time
            print(f"Shard {shard_id}: {fsize/1e6:.0f}MB -> GCS "
                  f"({total_written:,} total, {total_written/elapsed:.1f}/s)", flush=True)
            shard_id += 1

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - start_time
            rate = total_written / max(elapsed, 1)
            print(f"  [{i+1:,}] written={total_written:,} buffered={len(records)} "
                  f"skip={total_skipped} rate={rate:.1f}/s", flush=True)

    return shard_id, records, total_written, total_skipped, total_bytes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HF datasets to ArrayRecord")
    parser.add_argument("--dataset", required=True,
                        help="HF dataset name (e.g. 'diffusiondb', 'common-canvas/commoncatalog-cc-by')")
    parser.add_argument("--subset", default=None, help="Dataset subset/config (e.g. '2m_all')")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_folder", required=True, help="Local path or gs:// URL")
    parser.add_argument("--samples_per_shard", type=int, default=80000)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--tmp_dir", type=str, default="/tmp/hf_convert_tmp")
    parser.add_argument("--image_col", default="image", help="Image column name in HF dataset")
    parser.add_argument("--text_col", default="prompt", help="Text/caption column name")
    parser.add_argument("--workers", type=int, default=32,
                        help="Number of parallel workers for image processing")
    parser.add_argument("--zip_batch_size", type=int, default=50,
                        help="(DiffusionDB only) Zip files per batch")
    args = parser.parse_args()

    assert os.environ.get("HF_TOKEN"), "Set HF_TOKEN environment variable before running"

    print(f"=== Converting {args.dataset} -> ArrayRecord ===", flush=True)
    print(f"Output: {args.output_folder}, shard: {args.samples_per_shard}, "
          f"JPEG q{args.jpeg_quality}", flush=True)

    start = time.time()

    if args.dataset == "diffusiondb":
        # Direct zip batch download (2000 parts of ~633MB each)
        shard_id, records, total_written, total_skipped, total_bytes = convert_diffusiondb(args)
    else:
        # Batch parquet download for generic HF datasets
        shard_id, records, total_written, total_skipped, total_bytes = convert_hf_batch_download(args)

    # Flush remaining
    if records:
        fsize = flush_shard(shard_id, records, args.output_folder, args.tmp_dir)
        total_bytes += fsize
        shard_id += 1

    elapsed = time.time() - start
    print(f"\n=== DONE: {total_written:,} samples, {shard_id} shards, "
          f"{total_bytes/1e9:.1f}GB, {elapsed/60:.1f}min ({total_written/max(elapsed,1):.1f}/s) ===",
          flush=True)
