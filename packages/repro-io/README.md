# repro-io

Independent acquisition mechanisms; no study, engine, tracking or DB dependencies.

- `repro_io.http`: FetchResult, RangeFetcher, TokenBucketLimiter,
  SerialDownloader, BandwidthScheduler. Inject base URL and User-Agent.
- `repro_io.checksum.sha256_file`: streaming SHA-256.
- `repro_io.s3`: explicit S3Config and S3ObjectStore with SigV4 path addressing.
- `repro_io.commoncrawl.cdx`: CDX records, block locators, index reader and SURT.
- `repro_io.commoncrawl.fetcher.RangeFetcher`: Common Crawl endpoint/path adapter.
- `repro_io.archive`: ARCParser and ExtractedARCRecord, preserving parsing fallback.

Dependencies: requests and warcio. Source releases, normalization, filtering,
sampling, shard formats, credentials and environment interpretation belong to studies.
The F2 composition supplies its original User-Agent and transfer settings.

Independent test environment (from repository root):

```bash
uv run --isolated --no-project --with ./packages/repro-io --with pytest pytest packages/repro-io/tests -q
```
