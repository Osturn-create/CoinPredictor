#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime

import gbdt_pipeline as pipeline


def load_manifest(dataset_dir):
    manifest_path = os.path.join(dataset_dir, pipeline.SHARDED_DATASET_MANIFEST)
    with open(manifest_path, encoding="utf-8") as handle:
        return json.load(handle), manifest_path


def compatibility_signature(manifest):
    return pipeline.manifest_compatibility_signature(manifest)


def relative_path(base_dir, target_path):
    return os.path.relpath(os.path.abspath(target_path), os.path.abspath(base_dir))


def shard_inventory_entry(base_dir, shard):
    return {
        "symbol": shard["symbol"],
        "month": shard["month"],
        "csv_path": relative_path(base_dir, shard["csv_path"]),
        "meta_path": relative_path(base_dir, shard["meta_path"]),
        "compression": shard.get("compression", "none"),
        "row_count": int(shard.get("row_count", 0)),
    }


def merged_inventory_key(shard):
    return (
        str(shard.get("symbol", "")),
        str(shard.get("month", "")),
    )


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def backup_manifest(manifest_path):
    backup_path = manifest_path + ".pre_merge_" + timestamp() + ".bak"
    shutil.copy2(manifest_path, backup_path)
    return backup_path


def merged_manifest_template(source_manifest):
    merged = dict(source_manifest)
    merged["shards"] = []
    return merged


def manifest_inventory_shards(dataset_dir, dataset_manifest):
    dataset_dir = os.path.abspath(dataset_dir)
    listed_shards = dataset_manifest.get("shards")
    if not isinstance(listed_shards, list):
        raise ValueError("{} does not contain shard inventory entries".format(dataset_dir))
    shards = []
    for item in listed_shards:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", ""))
        month = str(item.get("month", ""))
        if not symbol or not month:
            continue
        csv_path = pipeline.resolve_dataset_relative_path(dataset_dir, item.get("csv_path"))
        meta_path = pipeline.resolve_dataset_relative_path(dataset_dir, item.get("meta_path"))
        compression = str(item.get("compression", "") or ("gzip" if str(csv_path).endswith(".gz") else "none"))
        csv_mtime_ns = 0
        csv_size = 0
        meta_mtime_ns = 0
        meta_size = 0
        if csv_path and os.path.exists(csv_path):
            csv_stat = os.stat(csv_path)
            csv_mtime_ns = getattr(csv_stat, "st_mtime_ns", int(csv_stat.st_mtime * 1000000000))
            csv_size = int(csv_stat.st_size)
        if meta_path and os.path.exists(meta_path):
            meta_stat = os.stat(meta_path)
            meta_mtime_ns = getattr(meta_stat, "st_mtime_ns", int(meta_stat.st_mtime * 1000000000))
            meta_size = int(meta_stat.st_size)
        shards.append({
            "symbol": symbol,
            "month": month,
            "csv_path": csv_path,
            "relative_csv_path": os.path.relpath(csv_path, dataset_dir) if csv_path else "",
            "meta_path": meta_path,
            "compression": compression,
            "row_count": int(item.get("row_count", 0)),
            "csv_mtime_ns": csv_mtime_ns,
            "csv_size": csv_size,
            "meta_mtime_ns": meta_mtime_ns,
            "meta_size": meta_size,
            "manifest_signature": item.get("manifest_signature", ""),
        })
    if not shards:
        raise ValueError("{} does not contain usable shard inventory entries".format(dataset_dir))
    return sorted(shards, key=lambda item: (item["symbol"], item["month"], item["csv_path"]))


def shard_content_sha1(shard):
    cached = shard.get("_content_sha1")
    if cached:
        return cached
    digest = hashlib.sha1()
    with pipeline.open_csv_text(shard["csv_path"]) as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk.encode("utf-8"))
    shard["_content_sha1"] = digest.hexdigest()
    return shard["_content_sha1"]


def equivalent_duplicate_shard(existing_shard, new_shard):
    return (
        int(existing_shard.get("row_count", 0)) == int(new_shard.get("row_count", 0))
        and str(existing_shard.get("manifest_signature", "")) == str(new_shard.get("manifest_signature", ""))
        and shard_content_sha1(existing_shard) == shard_content_sha1(new_shard)
    )


def duplicate_shard_conflict_message(existing_shard, new_shard):
    return (
        "conflicting duplicate shard for symbol={} month={}: {} and {} contain different data".format(
            existing_shard.get("symbol", ""),
            existing_shard.get("month", ""),
            existing_shard.get("csv_path", ""),
            new_shard.get("csv_path", ""),
        )
    )


def discover_or_inventory_shards(dataset_dir, dataset_manifest):
    try:
        return pipeline.discover_sharded_dataset_shards(dataset_dir, dataset_manifest)
    except ValueError:
        return manifest_inventory_shards(dataset_dir, dataset_manifest)


def collect_merged_entries(base_dir, dataset_dirs, expected_signature):
    shards_by_key = {}
    added_counts = []
    for dataset_dir in dataset_dirs:
        dataset_manifest, _ = load_manifest(dataset_dir)
        dataset_signature = compatibility_signature(dataset_manifest)
        if dataset_signature != expected_signature:
            raise ValueError(
                "{} is incompatible with the merged dataset. Feature/label configuration differs.".format(
                    dataset_dir,
                )
            )
        added = 0
        for shard in discover_or_inventory_shards(dataset_dir, dataset_manifest):
            key = merged_inventory_key({
                "symbol": shard["symbol"],
                "month": shard["month"],
            })
            existing = shards_by_key.get(key)
            if existing is not None:
                if not equivalent_duplicate_shard(existing["shard"], shard):
                    raise ValueError(duplicate_shard_conflict_message(existing["shard"], shard))
                continue
            shards_by_key[key] = {
                "entry": shard_inventory_entry(base_dir, shard),
                "shard": shard,
            }
            added += 1
        added_counts.append((dataset_dir, added))
    merged_entries = sorted(
        (item["entry"] for item in shards_by_key.values()),
        key=lambda item: (item["symbol"], item["month"], item["csv_path"]),
    )
    return merged_entries, added_counts


def ensure_clean_directory(path, replace=False):
    if os.path.exists(path):
        if not replace:
            raise ValueError(
                "{} already exists. Pass --replace-output to rebuild it intentionally.".format(path)
            )
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def create_combined_dataset(output_dataset_dir, source_dataset_dirs, replace_output=False):
    first_manifest, _ = load_manifest(source_dataset_dirs[0])
    expected_signature = compatibility_signature(first_manifest)
    merged_entries, added_counts = collect_merged_entries(
        output_dataset_dir,
        source_dataset_dirs,
        expected_signature,
    )
    ensure_clean_directory(output_dataset_dir, replace=replace_output)
    os.makedirs(os.path.join(output_dataset_dir, "shards"), exist_ok=True)
    merged_manifest = merged_manifest_template(first_manifest)
    merged_manifest["shards"] = merged_entries
    manifest_path = os.path.join(output_dataset_dir, pipeline.SHARDED_DATASET_MANIFEST)
    pipeline.atomic_write_json(manifest_path, merged_manifest)
    return {
        "manifest_path": manifest_path,
        "merged_shard_count": len(merged_entries),
        "source_added_counts": added_counts,
    }


def update_destination_manifest(destination_dataset_dir, source_dataset_dirs):
    destination_manifest, destination_manifest_path = load_manifest(destination_dataset_dir)
    expected_signature = compatibility_signature(destination_manifest)
    merged_entries, added_counts = collect_merged_entries(
        destination_dataset_dir,
        [destination_dataset_dir] + list(source_dataset_dirs),
        expected_signature,
    )
    existing_entries = destination_manifest.get("shards", [])
    if len(merged_entries) == len(existing_entries):
        return {
            "manifest_path": destination_manifest_path,
            "backup_path": "",
            "merged_shard_count": len(merged_entries),
            "source_added_counts": added_counts[1:],
        }
    updated_manifest = dict(destination_manifest)
    updated_manifest["shards"] = merged_entries
    backup_path = backup_manifest(destination_manifest_path)
    pipeline.atomic_write_json(destination_manifest_path, updated_manifest)
    return {
        "manifest_path": destination_manifest_path,
        "backup_path": backup_path,
        "merged_shard_count": len(merged_entries),
        "source_added_counts": added_counts[1:],
    }


def collect_source_cache_roots(source_dataset_dir, source_cache_dir, feature_storage):
    source_manifest, _ = load_manifest(source_dataset_dir)
    shards = discover_or_inventory_shards(source_dataset_dir, source_manifest)
    dtype = pipeline.np.float32 if feature_storage == "memmap32" else pipeline.np.float64
    shard_cache_dir = pipeline.sharded_dataset_cache_paths(
        source_dataset_dir,
        source_cache_dir,
        dtype,
    )["shard_cache_dir"]
    for shard in shards:
        paths = pipeline.shard_cache_paths(shard["csv_path"], shard_cache_dir, dtype)
        yield shard, paths["cache_root"]


def shard_seed_key(shard):
    return (
        str(shard.get("symbol", "")),
        str(shard.get("month", "")),
    )


def ensure_clean_cache_dir(cache_dir, replace=False):
    if os.path.exists(cache_dir):
        if not replace:
            return
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)


def seed_destination_shard_cache(destination_dataset_dir, destination_cache_dir,
                                 source_dataset_dirs, source_cache_dirs, feature_storage,
                                 replace_cache=False):
    if pipeline.np is None:
        raise ValueError("numpy is required for shard cache merging")
    dtype = pipeline.np.float32 if feature_storage == "memmap32" else pipeline.np.float64
    ensure_clean_cache_dir(destination_cache_dir, replace=replace_cache)
    destination_paths = pipeline.sharded_dataset_cache_paths(
        destination_dataset_dir,
        destination_cache_dir,
        dtype,
    )
    os.makedirs(destination_paths["shard_cache_dir"], exist_ok=True)
    copied = 0
    skipped = 0
    missing = 0
    duplicate_sources = 0
    seen_seed_keys = set()
    for source_dataset_dir, source_cache_dir in zip(source_dataset_dirs, source_cache_dirs):
        if not source_cache_dir:
            continue
        for shard, source_cache_root in collect_source_cache_roots(
                source_dataset_dir,
                source_cache_dir,
                feature_storage):
            seed_key = shard_seed_key(shard)
            if seed_key in seen_seed_keys:
                duplicate_sources += 1
                continue
            seen_seed_keys.add(seed_key)
            destination_cache_root = pipeline.shard_cache_paths(
                shard["csv_path"],
                destination_paths["shard_cache_dir"],
                dtype,
            )["cache_root"]
            if os.path.isdir(destination_cache_root):
                skipped += 1
                continue
            if not os.path.isdir(source_cache_root):
                missing += 1
                continue
            shutil.copytree(source_cache_root, destination_cache_root)
            copied += 1
    return {
        "copied": copied,
        "skipped": skipped,
        "missing": missing,
        "duplicate_sources": duplicate_sources,
        "destination_shard_cache_dir": destination_paths["shard_cache_dir"],
    }


def rebuild_aggregate_cache(dataset_dir, cache_dir, feature_storage, rebuild_cache=False):
    rows, feature_columns, has_returns = pipeline.load_rows(
        dataset_dir,
        feature_storage,
        cache_dir=cache_dir,
        rebuild_cache=rebuild_cache,
    )
    row_count = len(rows)
    if hasattr(rows, "cleanup"):
        rows.cleanup()
    return {
        "row_count": row_count,
        "feature_count": len(feature_columns),
        "has_returns": bool(has_returns),
    }


def rebuild_aggregate_cache_from_sources(dataset_dir, cache_dir, feature_storage,
                                         source_dataset_dirs, source_cache_dirs):
    if pipeline.np is None:
        raise ValueError("numpy is required for aggregate cache rebuild")
    dtype = pipeline.np.float32 if feature_storage == "memmap32" else pipeline.np.float64
    dataset_manifest, _ = load_manifest(dataset_dir)
    shards = discover_or_inventory_shards(dataset_dir, dataset_manifest)
    resolved_cache_dir = pipeline.resolve_cache_dir(dataset_dir, cache_dir)
    os.makedirs(resolved_cache_dir, exist_ok=True)
    paths = pipeline.sharded_dataset_cache_paths(dataset_dir, resolved_cache_dir, dtype)
    pipeline.clear_cache_files({
        "features": paths["features"],
        "metadata_arrays": paths["metadata_arrays"],
        "manifest": paths["manifest"],
    })
    os.makedirs(paths["shard_cache_dir"], exist_ok=True)

    loaded = []
    try:
        for source_dataset_dir, source_cache_dir in zip(source_dataset_dirs, source_cache_dirs):
            effective_cache_dir = source_cache_dir or resolved_cache_dir
            rows, feature_columns, has_returns = pipeline.load_rows(
                source_dataset_dir,
                feature_storage,
                cache_dir=effective_cache_dir,
                rebuild_cache=False,
            )
            loaded.append((source_dataset_dir, rows, list(feature_columns), bool(has_returns)))

        if not loaded:
            raise ValueError("no source datasets available for aggregate rebuild")

        feature_columns = loaded[0][2]
        has_returns = any(item[3] for item in loaded)
        for _, _, current_columns, _ in loaded[1:]:
            if current_columns != feature_columns:
                raise ValueError("source datasets have incompatible feature columns")

        total_rows = sum(len(item[1]) for item in loaded)
        symbols = []
        symbol_lookup = {}
        month_values = set()
        for _, rows, _, _ in loaded:
            table = rows.table
            for symbol_name in table.symbols:
                if symbol_name not in symbol_lookup:
                    symbol_lookup[symbol_name] = len(symbols)
                    symbols.append(symbol_name)
            month_values.update(str(month_name) for month_name in table.months if str(month_name).strip())
        months = sorted(month_values)
        month_lookup = {month_name: index for index, month_name in enumerate(months)}

        feature_values = pipeline.np.memmap(
            paths["features"],
            dtype=dtype,
            mode="w+",
            shape=(total_rows, len(feature_columns)),
        )
        symbol_codes = pipeline.np.empty(total_rows, dtype=pipeline.np.int32)
        month_codes = pipeline.np.empty(total_rows, dtype=pipeline.np.int32)
        month_indices = pipeline.np.empty(total_rows, dtype=pipeline.np.int16)
        open_times = pipeline.np.empty(total_rows, dtype=pipeline.np.int64)
        labels_values = pipeline.np.empty(total_rows, dtype=pipeline.np.int8)
        forward_returns = pipeline.np.empty(total_rows, dtype=pipeline.np.float32)
        trade_returns = pipeline.np.empty(total_rows, dtype=pipeline.np.float32)
        max_future_high_returns = pipeline.np.empty(total_rows, dtype=pipeline.np.float32)
        max_future_low_returns = pipeline.np.empty(total_rows, dtype=pipeline.np.float32)
        quote_volumes = pipeline.np.empty(total_rows, dtype=pipeline.np.float32)

        offset = 0
        for _, rows, _, _ in loaded:
            table = rows.table
            count = len(table.labels)
            feature_values[offset:offset + count, :] = table.features
            symbol_codes[offset:offset + count] = pipeline.np.asarray(
                [symbol_lookup[table.symbols[int(code)]] for code in table.symbol_codes],
                dtype=pipeline.np.int32,
            )
            global_month_codes = pipeline.np.asarray(
                [month_lookup[table.months[int(code)]] for code in table.month_codes],
                dtype=pipeline.np.int32,
            )
            month_codes[offset:offset + count] = global_month_codes
            month_indices[offset:offset + count] = global_month_codes.astype(pipeline.np.int16, copy=False)
            open_times[offset:offset + count] = table.open_times
            labels_values[offset:offset + count] = table.labels
            forward_returns[offset:offset + count] = table.forward_returns
            trade_returns[offset:offset + count] = table.trade_returns
            max_future_high_returns[offset:offset + count] = table.max_future_high_returns
            max_future_low_returns[offset:offset + count] = table.max_future_low_returns
            if table.quote_volumes is None:
                quote_volumes[offset:offset + count] = 0.0
            else:
                quote_volumes[offset:offset + count] = table.quote_volumes
            offset += count
        feature_values.flush()

        aggregate_table = pipeline.CompactTable(
            symbols,
            months,
            symbol_codes,
            month_codes,
            month_indices,
            open_times,
            labels_values,
            forward_returns,
            trade_returns,
            max_future_high_returns,
            max_future_low_returns,
            quote_volumes,
            feature_values,
            {name: index for index, name in enumerate(feature_columns)},
            paths["features"],
            False,
        )
        pipeline.write_metadata_arrays(paths, pipeline.metadata_arrays_from_table(aggregate_table))
        manifest = {
            "version": pipeline.SHARDED_AGGREGATE_CACHE_VERSION,
            "dataset_path": os.path.abspath(dataset_dir),
            "dataset_manifest_path": os.path.abspath(os.path.join(dataset_dir, pipeline.SHARDED_DATASET_MANIFEST)),
            "dataset_manifest_signature": pipeline.manifest_compatibility_signature(dataset_manifest),
            "inventory_signature": pipeline.sharded_inventory_signature(shards),
            "feature_dtype": pipeline.np.dtype(dtype).name,
            "feature_columns": feature_columns,
            "row_count": int(total_rows),
            "has_returns": bool(has_returns),
            "quote_volumes_present": True,
            "symbols": list(symbols),
            "months": list(months),
            "shard_count": len(shards),
            "shard_cache_hits": 0,
            "shard_cache_rebuilt": 0,
        }
        pipeline.atomic_write_json(paths["manifest"], manifest)
        return {
            "row_count": int(total_rows),
            "feature_count": len(feature_columns),
            "has_returns": bool(has_returns),
        }
    finally:
        for _, rows, _, _ in loaded:
            try:
                rows.cleanup()
            except Exception:
                pass


def print_added_counts(source_added_counts):
    for dataset_dir, added in source_added_counts:
        print(
            "Source {} contributed {} shard inventory entries".format(
                os.path.basename(dataset_dir),
                added,
            ),
            flush=True,
        )


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Build a fresh combined shard dataset manifest and cache, or merge extra shard inventories into an existing dataset.",
    )
    parser.add_argument("--destination-dataset",
                        help="Existing dataset manifest directory to update in place.")
    parser.add_argument("--output-dataset",
                        help="New lightweight dataset manifest directory to create from the union of all --source-dataset entries.")
    parser.add_argument("--destination-cache-dir", required=True)
    parser.add_argument("--source-dataset", action="append", dest="source_datasets", required=True,
                        help="Shard dataset directory to include in the merged dataset.")
    parser.add_argument("--source-cache-dir", action="append", dest="source_cache_dirs", default=[],
                        help="Optional cache root matching each --source-dataset.")
    parser.add_argument("--feature-storage", choices=("memmap32", "memmap64"), default="memmap32")
    parser.add_argument("--skip-cache-seed", action="store_true")
    parser.add_argument("--skip-aggregate-rebuild", action="store_true")
    parser.add_argument("--replace-output", action="store_true",
                        help="Delete and recreate --output-dataset and --destination-cache-dir before rebuilding.")
    return parser.parse_args(argv)


def validate_args(args):
    if bool(args.destination_dataset) == bool(args.output_dataset):
        raise ValueError("Specify exactly one of --destination-dataset or --output-dataset")
    if args.output_dataset and len(args.source_datasets) < 1:
        raise ValueError("--output-dataset requires at least one --source-dataset")
    if args.destination_dataset and len(args.source_datasets) < 1:
        raise ValueError("--destination-dataset requires at least one --source-dataset")


def normalize_source_cache_dirs(source_datasets, source_cache_dirs):
    normalized = list(source_cache_dirs)
    while len(normalized) < len(source_datasets):
        normalized.append("")
    if len(normalized) != len(source_datasets):
        raise ValueError("source cache configuration mismatch")
    return [os.path.abspath(path) if path else "" for path in normalized]


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    validate_args(args)

    source_dataset_dirs = [os.path.abspath(path) for path in args.source_datasets]
    source_cache_dirs = normalize_source_cache_dirs(source_dataset_dirs, args.source_cache_dirs)
    destination_cache_dir = os.path.abspath(args.destination_cache_dir)

    if args.output_dataset:
        output_dataset_dir = os.path.abspath(args.output_dataset)
        manifest_result = create_combined_dataset(
            output_dataset_dir,
            source_dataset_dirs,
            replace_output=args.replace_output,
        )
        dataset_dir = output_dataset_dir
        print("Created combined dataset manifest {}".format(manifest_result["manifest_path"]), flush=True)
        print_added_counts(manifest_result["source_added_counts"])
        print("Merged shard count={}".format(manifest_result["merged_shard_count"]), flush=True)
    else:
        destination_dataset_dir = os.path.abspath(args.destination_dataset)
        manifest_result = update_destination_manifest(
            destination_dataset_dir,
            source_dataset_dirs,
        )
        dataset_dir = destination_dataset_dir
        print("Updated {}".format(manifest_result["manifest_path"]), flush=True)
        if manifest_result.get("backup_path"):
            print("Backup saved to {}".format(manifest_result["backup_path"]), flush=True)
        print_added_counts(manifest_result["source_added_counts"])
        print("Merged shard count={}".format(manifest_result["merged_shard_count"]), flush=True)

    if not args.skip_cache_seed:
        seed_result = seed_destination_shard_cache(
            dataset_dir,
            destination_cache_dir,
            source_dataset_dirs,
            source_cache_dirs,
            args.feature_storage,
            replace_cache=args.replace_output,
        )
        print(
            "Seeded shard cache roots: copied={} skipped={} missing={} duplicate_sources={}".format(
                seed_result["copied"],
                seed_result["skipped"],
                seed_result["missing"],
                seed_result["duplicate_sources"],
            ),
            flush=True,
        )
        print("Destination shard cache dir={}".format(seed_result["destination_shard_cache_dir"]), flush=True)
    else:
        ensure_clean_cache_dir(destination_cache_dir, replace=args.replace_output)
        print("Skipped shard cache seeding.", flush=True)

    if not args.skip_aggregate_rebuild:
        try:
            aggregate_result = rebuild_aggregate_cache(
                dataset_dir,
                destination_cache_dir,
                args.feature_storage,
                rebuild_cache=args.replace_output,
            )
        except ValueError as error:
            print(
                "Standard aggregate rebuild unavailable ({}); falling back to source-cache aggregate merge.".format(
                    error
                ),
                flush=True,
            )
            aggregate_result = rebuild_aggregate_cache_from_sources(
                dataset_dir,
                destination_cache_dir,
                args.feature_storage,
                source_dataset_dirs,
                source_cache_dirs,
            )
        print(
            "Aggregate cache rebuilt: rows={} features={} has_returns={}".format(
                aggregate_result["row_count"],
                aggregate_result["feature_count"],
                int(aggregate_result["has_returns"]),
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
