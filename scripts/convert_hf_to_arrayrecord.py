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


def image_to_jpeg(image_bytes_or_pil, quality=95):
    """Convert any image (PIL or bytes) to JPEG bytes via cv2."""
    if hasattr(image_bytes_or_pil, 'save'):
        # PIL Image
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
    """DiffusionDB stores images as zip files. Download in batches, convert, upload."""
    from huggingface_hub import hf_hub_download
    import shutil

    REPO = "poloclub/diffusiondb"
    NUM_PARTS = 2000  # 2M images across 2000 zip parts

    shard_id, records, total_written, total_skipped, total_bytes = 0, [], 0, 0, 0
    start_time = time.time()

    for batch_start in range(1, NUM_PARTS + 1, args.zip_batch_size):
        batch_end = min(batch_start + args.zip_batch_size, NUM_PARTS + 1)
        print(f"\n--- Parts {batch_start}-{batch_end-1} ---", flush=True)

        for part_id in range(batch_start, batch_end):
            part_name = f"part-{part_id:06d}"
            try:
                zip_path = hf_hub_download(repo_id=REPO, repo_type="dataset",
                    filename=f"images/{part_name}/{part_name}.zip", cache_dir=args.tmp_dir)
                json_path = hf_hub_download(repo_id=REPO, repo_type="dataset",
                    filename=f"images/{part_name}/{part_name}.json", cache_dir=args.tmp_dir)
            except Exception as e:
                print(f"  Skip {part_name}: {e}", flush=True)
                continue

            try:
                with open(json_path) as f:
                    metadata = json.load(f)
            except Exception:
                metadata = {}

            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for img_name in zf.namelist():
                        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                            continue
                        try:
                            jpg_bytes = image_to_jpeg(zf.read(img_name), quality=args.jpeg_quality)
                            if jpg_bytes is None:
                                total_skipped += 1
                                continue
                            meta_entry = metadata.get(img_name, {})
                            caption = str(meta_entry.get("p", ""))
                            meta = {k: meta_entry[k] for k in ["se", "c", "st", "sa"] if k in meta_entry}

                            records.append(pack_dict_of_byte_arrays({
                                "key": f"diffdb_{part_id}_{img_name}".encode("utf-8"),
                                "jpg": jpg_bytes,
                                "txt": caption.encode("utf-8"),
                                "meta": json.dumps(meta).encode("utf-8"),
                            }))
                            total_written += 1

                            if len(records) >= args.samples_per_shard:
                                fsize = flush_shard(shard_id, records, args.output_folder, args.tmp_dir)
                                total_bytes += fsize
                                elapsed = time.time() - start_time
                                print(f"  Shard {shard_id}: {fsize/1e6:.0f}MB -> GCS "
                                      f"({total_written:,} total, {total_written/elapsed:.1f}/s)", flush=True)
                                shard_id += 1
                                records = []
                        except Exception:
                            total_skipped += 1
            except Exception as e:
                print(f"  Zip error {part_name}: {e}", flush=True)

        # Clean HF cache for this batch
        cache_path = os.path.join(args.tmp_dir, "datasets--poloclub--diffusiondb")
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path, ignore_errors=True)

        elapsed = time.time() - start_time
        rate = total_written / max(elapsed, 1)
        eta_h = (2_000_000 - total_written) / max(rate, 0.1) / 3600
        print(f"  Batch done: {total_written:,} written, {total_skipped} skipped, "
              f"{rate:.1f}/s, ETA {eta_h:.1f}h", flush=True)

    return shard_id, records, total_written, total_skipped, total_bytes


# ---- Generic HF dataset: streaming ----

def convert_hf_streaming(args):
    """Generic HF dataset with image + text columns. Streams one sample at a time."""
    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.subset, split=args.split,
                      streaming=True, trust_remote_code=True)

    shard_id, records, total_written, total_skipped, total_bytes = 0, [], 0, 0, 0
    start_time = time.time()

    for i, sample in enumerate(ds):
        try:
            img = sample.get(args.image_col)
            if img is None:
                total_skipped += 1
                continue
            jpg_bytes = image_to_jpeg(img, quality=args.jpeg_quality)
            if jpg_bytes is None:
                total_skipped += 1
                continue
            caption = str(sample.get(args.text_col, ""))
            meta = {}
            for k in ["seed", "step", "cfg", "sampler", "width", "height", "key", "sha256"]:
                v = sample.get(k)
                if v is not None and isinstance(v, (int, float, str, bool)):
                    meta[k] = v

            records.append(pack_dict_of_byte_arrays({
                "key": f"{args.dataset.split('/')[-1]}_{i}".encode("utf-8"),
                "jpg": jpg_bytes,
                "txt": caption.encode("utf-8"),
                "meta": json.dumps(meta).encode("utf-8"),
            }))
            total_written += 1

            if len(records) >= args.samples_per_shard:
                fsize = flush_shard(shard_id, records, args.output_folder, args.tmp_dir)
                total_bytes += fsize
                elapsed = time.time() - start_time
                print(f"Shard {shard_id}: {fsize/1e6:.0f}MB -> GCS "
                      f"({total_written:,} total, {total_written/elapsed:.1f}/s)", flush=True)
                shard_id += 1
                records = []

            if (i + 1) % 5000 == 0:
                elapsed = time.time() - start_time
                processed = total_written + len(records)
                rate = processed / max(elapsed, 1)
                print(f"  [{i+1:,}] written={total_written:,} buffered={len(records)} "
                      f"skip={total_skipped} rate={rate:.1f}/s", flush=True)

        except Exception:
            total_skipped += 1

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
    parser.add_argument("--zip_batch_size", type=int, default=50,
                        help="(DiffusionDB only) Zip files per batch")
    args = parser.parse_args()

    assert os.environ.get("HF_TOKEN"), "Set HF_TOKEN environment variable before running"

    print(f"=== Converting {args.dataset} -> ArrayRecord ===", flush=True)
    print(f"Output: {args.output_folder}, shard: {args.samples_per_shard}, "
          f"JPEG q{args.jpeg_quality}", flush=True)

    start = time.time()

    if args.dataset == "diffusiondb":
        args.subset = args.subset or "2m_all"
        shard_id, records, total_written, total_skipped, total_bytes = convert_diffusiondb(args)
    else:
        shard_id, records, total_written, total_skipped, total_bytes = convert_hf_streaming(args)

    # Flush remaining
    if records:
        fsize = flush_shard(shard_id, records, args.output_folder, args.tmp_dir)
        total_bytes += fsize
        shard_id += 1

    elapsed = time.time() - start
    print(f"\n=== DONE: {total_written:,} samples, {shard_id} shards, "
          f"{total_bytes/1e9:.1f}GB, {elapsed/60:.1f}min ({total_written/max(elapsed,1):.1f}/s) ===",
          flush=True)
