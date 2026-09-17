# F2 Common Crawl producer

`f2_cc` is the independent Common Crawl producer for F2. It acquires ARC
records, extracts and filters English news text, and emits the completed corpus
that F2 consumes. It does not belong to the F2 catalog or corpus-release
database.

## Next collection policy

The previous 10K/50K runs were validation runs only. Their candidate records,
processing rows, audit assignments, ARC locations, provenance, and temporary
outputs are not inputs to the final collection. They may be discarded after
the validation statistics have been archived.

The next collection targets **40 billion retained words**. Collection should
be streaming and bounded:

1. discover and fetch one ARC range;
2. parse, extract, classify, and filter it immediately;
3. append only accepted clean text to the current shard;
4. discard the ARC bytes and rejected document as soon as processing finishes;
5. rotate shards at a fixed size and update the retained-word counter;
6. stop when the committed clean-text total reaches 40B words.

The final producer storage contains clean-text shards and the smallest
manifest needed to identify shard order, word totals, byte sizes, and checksums.
It does not retain raw ARC payloads, candidate URLs, per-document processing
metadata, audit rows, or full provenance. Staging is limited to the current
range, parser buffers, and one shard being written; it must not mirror the
collection or accumulate per-document files.

## Capacity and operation

The 40B target is a retained-word target, not a number of sampled candidates.
The validation runs suggest roughly 100M candidate records may be needed, but
the production run must recalibrate this after a small pilot. Final clean text
is expected to fit comfortably below 1TB with compressed shards; retaining ARC
downloads would require hundreds of TB and is intentionally out of scope.

The producer must checkpoint only at shard boundaries. A restart may repeat the
current uncommitted shard, but must never require the old validation database
or a historical candidate/provenance store. F2 imports a completed release
manifest only after all shards have passed checksum, size, ordering, and total
word-count validation.

The existing validation data is therefore evidence for planning, not a source
corpus. A fresh production run starts with a new run ID and a new release
manifest.
