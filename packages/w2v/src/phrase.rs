//! Multi-threaded, hash-partitioned, bounded-memory phrase detection and corpus rewriting.

use crate::config::Status;
use std::{
    cmp::Ordering as CmpOrdering,
    collections::{BinaryHeap, HashMap, HashSet},
    fs::{self, File},
    hash::{BuildHasherDefault, Hasher},
    io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    thread,
};

const NUM_PARTITIONS: usize = 64;
const UNIGRAM_MEMORY_BUDGET: usize = 256 * 1024 * 1024;
const BIGRAM_MEMORY_BUDGET: usize = 2048 * 1024 * 1024;
const IO_BUFFER_BYTES: usize = 256 * 1024;
const MIN_PARALLEL_BYTES: u64 = 256 * 1024;
const MAX_THREADS: usize = 20;

#[derive(Default)]
struct FxHasher(u64);

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        const K: u64 = 0x517cc1b727220a95;
        let mut h = self.0;
        while bytes.len() >= 8 {
            let word = u64::from_ne_bytes(bytes[..8].try_into().unwrap());
            h = (h.rotate_left(5) ^ word).wrapping_mul(K);
            bytes = &bytes[8..];
        }
        for &b in bytes {
            h = (h.rotate_left(5) ^ u64::from(b)).wrapping_mul(K);
        }
        self.0 = h;
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.0 = (self.0.rotate_left(5) ^ (i as u64)).wrapping_mul(0x517cc1b727220a95);
    }
}

#[inline(always)]
fn partition_of(bytes: &[u8]) -> usize {
    let mut hasher = FxHasher(0);
    hasher.write(bytes);
    (hasher.finish() as usize) & (NUM_PARTITIONS - 1)
}

type FastHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;
type FastHashSet<K> = HashSet<K, BuildHasherDefault<FxHasher>>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhrasePassReport {
    pub input_tokens: u64,
    pub documents: u64,
    pub unique_unigrams: u64,
    pub unique_bigrams: u64,
    pub joined_count: u64,
    pub output_tokens: u64,
}

#[inline(always)]
fn is_whitespace(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | b'\n' | 0x0b | 0x0c)
}

#[inline(always)]
fn validate_token(token: &[u8]) -> Result<(), Status> {
    if token == b"</s>" || token.contains(&0) {
        return Err(Status::InvalidArgument);
    }
    Ok(())
}

fn align_to_newline(path: &Path, offset: u64, file_len: u64) -> Result<u64, Status> {
    if offset == 0 || offset >= file_len {
        return Ok(offset.min(file_len));
    }
    let mut file = File::open(path).map_err(|_| Status::IoError)?;
    file.seek(SeekFrom::Start(offset - 1))
        .map_err(|_| Status::IoError)?;
    let mut prev = [0u8; 1];
    file.read_exact(&mut prev).map_err(|_| Status::IoError)?;
    if prev[0] == b'\n' {
        return Ok(offset);
    }
    let mut reader = BufReader::with_capacity(64 * 1024, file);
    let mut pos = offset;
    loop {
        let buf = reader.fill_buf().map_err(|_| Status::IoError)?;
        if buf.is_empty() {
            return Ok(pos.min(file_len));
        }
        if let Some(idx) = buf.iter().position(|&b| b == b'\n') {
            pos += (idx + 1) as u64;
            return Ok(pos.min(file_len));
        }
        let consumed = buf.len();
        pos += consumed as u64;
        reader.consume(consumed);
    }
}

fn compute_shards(
    path: &Path,
    workers: usize,
) -> Result<(u64, bool, Vec<(u64, u64)>), Status> {
    let mut file = File::open(path).map_err(|_| Status::IoError)?;
    let file_len = file.seek(SeekFrom::End(0)).map_err(|_| Status::IoError)?;
    if file_len == 0 {
        return Ok((0, false, Vec::new()));
    }
    file.seek(SeekFrom::Start(file_len - 1))
        .map_err(|_| Status::IoError)?;
    let mut last_byte = [0u8; 1];
    file.read_exact(&mut last_byte)
        .map_err(|_| Status::IoError)?;
    let ends_without_lf = last_byte[0] != b'\n';

    let effective_workers = if file_len < MIN_PARALLEL_BYTES {
        1
    } else {
        workers.max(1)
    };

    let mut shards = Vec::with_capacity(effective_workers);
    let mut prev_end = 0u64;
    for w in 0..effective_workers {
        let start = if w == 0 {
            0
        } else {
            let raw = ((file_len as u128 * w as u128) / effective_workers as u128) as u64;
            align_to_newline(path, raw.max(prev_end), file_len)?
        };
        let end = if w + 1 == effective_workers {
            file_len
        } else {
            let raw = ((file_len as u128 * (w + 1) as u128) / effective_workers as u128) as u64;
            align_to_newline(path, raw.max(start), file_len)?
        };
        if start < end {
            shards.push((start, end));
            prev_end = end;
        }
    }
    Ok((file_len, ends_without_lf, shards))
}

fn scan_shard<F>(
    path: &Path,
    start: u64,
    end: u64,
    is_last_shard_without_lf: bool,
    chunk_bytes: usize,
    abort: &AtomicBool,
    mut on_event: F,
) -> Result<(), Status>
where
    F: FnMut(Option<&[u8]>) -> Result<(), Status>,
{
    let mut file = File::open(path).map_err(|_| Status::IoError)?;
    file.seek(SeekFrom::Start(start))
        .map_err(|_| Status::IoError)?;
    let mut remaining = (end - start) as usize;
    let step = chunk_bytes.max(1);
    let mut buf = vec![0u8; step.min(remaining.max(1))];
    let mut pending: Vec<u8> = Vec::new();

    while remaining > 0 {
        if abort.load(Ordering::Relaxed) {
            return Err(Status::IoError);
        }
        let to_read = buf.len().min(remaining);
        let n = file.read(&mut buf[..to_read]).map_err(|_| Status::IoError)?;
        if n == 0 {
            return Err(Status::IoError);
        }
        remaining -= n;
        let chunk = &buf[..n];
        let mut i = 0;
        while i < n {
            if !is_whitespace(chunk[i]) {
                let start_tok = i;
                i += 1;
                while i < n && !is_whitespace(chunk[i]) {
                    i += 1;
                }
                if pending.is_empty() && i < n {
                    let tok = &chunk[start_tok..i];
                    validate_token(tok)?;
                    on_event(Some(tok))?;
                } else {
                    pending.extend_from_slice(&chunk[start_tok..i]);
                    if i < n {
                        validate_token(&pending)?;
                        on_event(Some(&pending))?;
                        pending.clear();
                    }
                }
            } else {
                if !pending.is_empty() {
                    validate_token(&pending)?;
                    on_event(Some(&pending))?;
                    pending.clear();
                }
                while i < n && is_whitespace(chunk[i]) {
                    if chunk[i] == b'\n' {
                        on_event(None)?;
                    }
                    i += 1;
                }
            }
        }
    }

    if !pending.is_empty() {
        validate_token(&pending)?;
        on_event(Some(&pending))?;
        pending.clear();
    }
    if is_last_shard_without_lf {
        on_event(None)?;
    }
    Ok(())
}

/// One spilled file containing `NUM_PARTITIONS` contiguous sorted sections.
/// No file descriptor is kept open after spilling.
struct PartitionedDiskRun {
    path: PathBuf,
    offsets: [u64; NUM_PARTITIONS + 1],
}

fn spill_partitions_to_disk(
    partitions: &mut [FastHashMap<Box<[u8]>, u64>],
    path: PathBuf,
) -> Result<PartitionedDiskRun, Status> {
    let file = File::create(&path).map_err(|_| Status::IoError)?;
    let mut writer = BufWriter::with_capacity(IO_BUFFER_BYTES, file);
    let mut offsets = [0u64; NUM_PARTITIONS + 1];
    let mut current_offset = 0u64;

    for (p, map) in partitions.iter_mut().enumerate() {
        offsets[p] = current_offset;
        let mut entries: Vec<(Box<[u8]>, u64)> = map.drain().collect();
        entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        for (key, count) in entries {
            let len = u32::try_from(key.len()).map_err(|_| Status::InvalidArgument)?;
            writer
                .write_all(&len.to_le_bytes())
                .map_err(|_| Status::IoError)?;
            writer.write_all(&key).map_err(|_| Status::IoError)?;
            writer
                .write_all(&count.to_le_bytes())
                .map_err(|_| Status::IoError)?;
            current_offset += 4 + key.len() as u64 + 8;
        }
    }
    offsets[NUM_PARTITIONS] = current_offset;
    writer.flush().map_err(|_| Status::IoError)?;
    Ok(PartitionedDiskRun { path, offsets })
}

enum PartitionRunCursor {
    Memory(std::vec::IntoIter<(Box<[u8]>, u64)>),
    Disk {
        reader: BufReader<File>,
        remaining_bytes: u64,
    },
}

impl PartitionRunCursor {
    fn open_disk_section(run: &PartitionedDiskRun, partition: usize) -> Result<Option<Self>, Status> {
        let start = run.offsets[partition];
        let end = run.offsets[partition + 1];
        if start >= end {
            return Ok(None);
        }
        let mut file = File::open(&run.path).map_err(|_| Status::IoError)?;
        file.seek(SeekFrom::Start(start))
            .map_err(|_| Status::IoError)?;
        Ok(Some(PartitionRunCursor::Disk {
            reader: BufReader::with_capacity(32 * 1024, file),
            remaining_bytes: end - start,
        }))
    }

    fn next_item(&mut self) -> Result<Option<(Box<[u8]>, u64)>, Status> {
        match self {
            PartitionRunCursor::Memory(iter) => Ok(iter.next()),
            PartitionRunCursor::Disk {
                reader,
                remaining_bytes,
            } => {
                if *remaining_bytes == 0 {
                    return Ok(None);
                }
                let mut len_buf = [0u8; 4];
                reader
                    .read_exact(&mut len_buf)
                    .map_err(|_| Status::IoError)?;
                let key_len = u32::from_le_bytes(len_buf) as usize;
                let mut key = vec![0u8; key_len];
                reader.read_exact(&mut key).map_err(|_| Status::IoError)?;
                let mut count_buf = [0u8; 8];
                reader
                    .read_exact(&mut count_buf)
                    .map_err(|_| Status::IoError)?;
                let record_bytes = 4 + key_len as u64 + 8;
                *remaining_bytes = remaining_bytes.saturating_sub(record_bytes);
                let count = u64::from_le_bytes(count_buf);
                Ok(Some((key.into_boxed_slice(), count)))
            }
        }
    }
}

struct HeapItem {
    key: Box<[u8]>,
    count: u64,
    run_idx: usize,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.run_idx == other.run_idx
    }
}
impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        other
            .key
            .cmp(&self.key)
            .then_with(|| other.run_idx.cmp(&self.run_idx))
    }
}

struct MergeIter {
    runs: Vec<PartitionRunCursor>,
    heap: BinaryHeap<HeapItem>,
}

impl MergeIter {
    fn new(mut runs: Vec<PartitionRunCursor>) -> Result<Self, Status> {
        let mut heap = BinaryHeap::with_capacity(runs.len());
        for (idx, run) in runs.iter_mut().enumerate() {
            if let Some((key, count)) = run.next_item()? {
                heap.push(HeapItem {
                    key,
                    count,
                    run_idx: idx,
                });
            }
        }
        Ok(Self { runs, heap })
    }

    fn next_merged(&mut self) -> Result<Option<(Box<[u8]>, u64)>, Status> {
        let Some(first) = self.heap.pop() else {
            return Ok(None);
        };
        let key = first.key;
        let mut total_count = first.count;
        if let Some((next_key, next_count)) = self.runs[first.run_idx].next_item()? {
            self.heap.push(HeapItem {
                key: next_key,
                count: next_count,
                run_idx: first.run_idx,
            });
        }
        while let Some(top) = self.heap.peek() {
            if top.key != key {
                break;
            }
            let same = self.heap.pop().unwrap();
            total_count += same.count;
            if let Some((next_key, next_count)) = self.runs[same.run_idx].next_item()? {
                self.heap.push(HeapItem {
                    key: next_key,
                    count: next_count,
                    run_idx: same.run_idx,
                });
            }
        }
        Ok(Some((key, total_count)))
    }
}

struct WorkerCountResult {
    tokens: u64,
    docs: u64,
    unigram_mem: Vec<FastHashMap<Box<[u8]>, u64>>,
    bigram_mem: Vec<FastHashMap<Box<[u8]>, u64>>,
    unigram_disk: Vec<PartitionedDiskRun>,
    bigram_disk: Vec<PartitionedDiskRun>,
}

#[allow(clippy::too_many_arguments)]
fn count_worker<FProgress>(
    worker_id: usize,
    source: &Path,
    start: u64,
    end: u64,
    is_last_without_lf: bool,
    work_dir: &Path,
    chunk_bytes: usize,
    unigram_budget: usize,
    bigram_budget: usize,
    global_docs: &AtomicU64,
    abort: &AtomicBool,
    progress: &FProgress,
) -> Result<WorkerCountResult, Status>
where
    FProgress: Fn(u64) -> Result<(), Status>,
{
    let mut unigram_parts: Vec<FastHashMap<Box<[u8]>, u64>> =
        (0..NUM_PARTITIONS).map(|_| FastHashMap::default()).collect();
    let mut bigram_parts: Vec<FastHashMap<Box<[u8]>, u64>> =
        (0..NUM_PARTITIONS).map(|_| FastHashMap::default()).collect();
    let mut unigram_bytes = 0usize;
    let mut bigram_bytes = 0usize;
    let mut unigram_disk = Vec::new();
    let mut bigram_disk = Vec::new();
    let mut uni_seq = 0usize;
    let mut bi_seq = 0usize;

    let mut local_tokens = 0u64;
    let mut local_docs = 0u64;
    let mut prev_token: Vec<u8> = Vec::new();
    let mut has_prev = false;
    let mut scratch_bigram: Vec<u8> = Vec::new();

    scan_shard(
        source,
        start,
        end,
        is_last_without_lf,
        chunk_bytes,
        abort,
        |event| match event {
            Some(tok) => {
                local_tokens += 1;
                let u_part = &mut unigram_parts[partition_of(tok)];
                if let Some(c) = u_part.get_mut(tok) {
                    *c += 1;
                } else {
                    unigram_bytes += tok.len() + 32;
                    u_part.insert(tok.into(), 1);
                }
                if has_prev {
                    scratch_bigram.clear();
                    scratch_bigram.reserve(prev_token.len() + 1 + tok.len());
                    scratch_bigram.extend_from_slice(&prev_token);
                    scratch_bigram.push(0);
                    scratch_bigram.extend_from_slice(tok);
                    let b_part = &mut bigram_parts[partition_of(&scratch_bigram)];
                    if let Some(c) = b_part.get_mut(scratch_bigram.as_slice()) {
                        *c += 1;
                    } else {
                        bigram_bytes += scratch_bigram.len() + 32;
                        b_part.insert(scratch_bigram.as_slice().into(), 1);
                    }
                }
                prev_token.clear();
                prev_token.extend_from_slice(tok);
                has_prev = true;

                if unigram_bytes >= unigram_budget {
                    let run_path = work_dir.join(format!("uni-{worker_id}-{uni_seq}.prun"));
                    uni_seq += 1;
                    unigram_disk.push(spill_partitions_to_disk(&mut unigram_parts, run_path)?);
                    unigram_bytes = 0;
                }
                if bigram_bytes >= bigram_budget {
                    let run_path = work_dir.join(format!("bi-{worker_id}-{bi_seq}.prun"));
                    bi_seq += 1;
                    bigram_disk.push(spill_partitions_to_disk(&mut bigram_parts, run_path)?);
                    bigram_bytes = 0;
                }
                Ok(())
            }
            None => {
                has_prev = false;
                local_docs += 1;
                let doc_num = global_docs.fetch_add(1, Ordering::Relaxed) + 1;
                if doc_num.is_multiple_of(100_000) {
                    progress(doc_num)?;
                }
                Ok(())
            }
        },
    )?;

    Ok(WorkerCountResult {
        tokens: local_tokens,
        docs: local_docs,
        unigram_mem: unigram_parts,
        bigram_mem: bigram_parts,
        unigram_disk,
        bigram_disk,
    })
}

/// Merge a single partition across all in-memory maps and spilled disk runs.
/// When `disk_runs.is_empty()`, merges directly into a single hash map without sorting.
fn merge_partition<F>(
    partition: usize,
    mem_maps: Vec<FastHashMap<Box<[u8]>, u64>>,
    disk_runs: &[PartitionedDiskRun],
    mut on_item: F,
) -> Result<(), Status>
where
    F: FnMut(Box<[u8]>, u64) -> Result<(), Status>,
{
    if disk_runs.is_empty() {
        let mut combined: FastHashMap<Box<[u8]>, u64> = FastHashMap::default();
        for mut map in mem_maps {
            if combined.is_empty() {
                combined = map;
            } else {
                for (k, v) in map.drain() {
                    *combined.entry(k).or_insert(0) += v;
                }
            }
        }
        for (k, v) in combined {
            on_item(k, v)?;
        }
        return Ok(());
    }

    let mut cursors = Vec::with_capacity(mem_maps.len() + disk_runs.len());
    for mut map in mem_maps {
        if !map.is_empty() {
            let mut entries: Vec<(Box<[u8]>, u64)> = map.drain().collect();
            entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            cursors.push(PartitionRunCursor::Memory(entries.into_iter()));
        }
    }
    for disk_run in disk_runs {
        if let Some(cursor) = PartitionRunCursor::open_disk_section(disk_run, partition)? {
            cursors.push(cursor);
        }
    }
    let mut merger = MergeIter::new(cursors)?;
    while let Some((key, count)) = merger.next_merged()? {
        on_item(key, count)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_worker<W: Write, FProgress>(
    source: &Path,
    start: u64,
    end: u64,
    is_last_without_lf: bool,
    chunk_bytes: usize,
    separator: &[u8],
    eligible_partitions: &[FastHashSet<Box<[u8]>>],
    has_any_eligible: bool,
    mut writer: W,
    global_written_docs: &AtomicU64,
    abort: &AtomicBool,
    progress: &FProgress,
) -> Result<(u64, W), Status>
where
    FProgress: Fn(u64) -> Result<(), Status>,
{
    let mut joined = 0u64;
    let mut left: Vec<u8> = Vec::new();
    let mut has_left = false;
    let mut first = true;
    let mut scratch_pair: Vec<u8> = Vec::new();

    let emit = |out: &mut W, first_flag: &mut bool, tok: &[u8]| -> Result<(), Status> {
        if !*first_flag {
            out.write_all(b" ").map_err(|_| Status::IoError)?;
        }
        out.write_all(tok).map_err(|_| Status::IoError)?;
        *first_flag = false;
        Ok(())
    };

    scan_shard(
        source,
        start,
        end,
        is_last_without_lf,
        chunk_bytes,
        abort,
        |event| match event {
            Some(tok) => {
                if !has_left {
                    left.clear();
                    left.extend_from_slice(tok);
                    has_left = true;
                    return Ok(());
                }
                let is_eligible = if has_any_eligible {
                    scratch_pair.clear();
                    scratch_pair.reserve(left.len() + 1 + tok.len());
                    scratch_pair.extend_from_slice(&left);
                    scratch_pair.push(0);
                    scratch_pair.extend_from_slice(tok);
                    let part = partition_of(&scratch_pair);
                    eligible_partitions[part].contains(scratch_pair.as_slice())
                } else {
                    false
                };
                if is_eligible {
                    if left.len() + separator.len() + tok.len() == 4 {
                        let mut candidate = [0u8; 4];
                        candidate[..left.len()].copy_from_slice(&left);
                        candidate[left.len()..left.len() + separator.len()]
                            .copy_from_slice(separator);
                        candidate[left.len() + separator.len()..].copy_from_slice(tok);
                        if &candidate == b"</s>" {
                            return Err(Status::InvalidArgument);
                        }
                    }
                    if !first {
                        writer.write_all(b" ").map_err(|_| Status::IoError)?;
                    }
                    writer.write_all(&left).map_err(|_| Status::IoError)?;
                    writer.write_all(separator).map_err(|_| Status::IoError)?;
                    writer.write_all(tok).map_err(|_| Status::IoError)?;
                    first = false;
                    joined += 1;
                    has_left = false;
                } else {
                    emit(&mut writer, &mut first, &left)?;
                    left.clear();
                    left.extend_from_slice(tok);
                }
                Ok(())
            }
            None => {
                if has_left {
                    emit(&mut writer, &mut first, &left)?;
                    has_left = false;
                }
                writer.write_all(b"\n").map_err(|_| Status::IoError)?;
                first = true;
                let doc_num = global_written_docs.fetch_add(1, Ordering::Relaxed) + 1;
                if doc_num.is_multiple_of(100_000) {
                    progress(doc_num)?;
                }
                Ok(())
            }
        },
    )?;

    writer.flush().map_err(|_| Status::IoError)?;
    Ok((joined, writer))
}

#[allow(clippy::too_many_arguments)]
pub fn run_phrase_pass<FCountProgress, FWriteProgress>(
    source: &Path,
    destination: &Path,
    work_dir: &Path,
    min_count: u64,
    threshold: f64,
    separator: &[u8],
    chunk_bytes: usize,
    threads: usize,
    count_progress: FCountProgress,
    write_progress: FWriteProgress,
) -> Result<PhrasePassReport, Status>
where
    FCountProgress: Fn(u64) -> Result<(), Status> + Sync,
    FWriteProgress: Fn(u64, u64) -> Result<(), Status> + Sync,
{
    if min_count < 1 || !threshold.is_finite() || threshold < 0.0 || separator.is_empty() {
        return Err(Status::InvalidArgument);
    }
    if separator.contains(&0) || separator.iter().any(|&b| is_whitespace(b)) {
        return Err(Status::InvalidArgument);
    }

    let requested_threads = if chunk_bytes < 4096 {
        1
    } else if threads == 0 {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(MAX_THREADS)
    } else {
        threads.min(MAX_THREADS)
    };

    let (file_len, ends_without_lf, shards) = compute_shards(source, requested_threads)?;
    if shards.is_empty() {
        let out = File::create(destination).map_err(|_| Status::IoError)?;
        out.sync_all().map_err(|_| Status::IoError)?;
        return Ok(PhrasePassReport {
            input_tokens: 0,
            documents: 0,
            unique_unigrams: 0,
            unique_bigrams: 0,
            joined_count: 0,
            output_tokens: 0,
        });
    }

    let active_workers = shards.len();
    let unigram_budget = (UNIGRAM_MEMORY_BUDGET / active_workers).max(256 * 1024);
    let bigram_budget = (BIGRAM_MEMORY_BUDGET / active_workers).max(1024 * 1024);
    let global_docs = AtomicU64::new(0);
    let abort = AtomicBool::new(false);

    // Phase 1: Parallel Shard Counting
    let count_results: Vec<Result<WorkerCountResult, Status>> = if active_workers == 1 {
        let (start, end) = shards[0];
        vec![count_worker(
            0,
            source,
            start,
            end,
            end == file_len && ends_without_lf,
            work_dir,
            chunk_bytes,
            unigram_budget,
            bigram_budget,
            &global_docs,
            &abort,
            &count_progress,
        )]
    } else {
        thread::scope(|s| {
            let mut handles = Vec::with_capacity(active_workers);
            for (idx, &(start, end)) in shards.iter().enumerate() {
                let is_last_without_lf = end == file_len && ends_without_lf;
                let global_docs_ref = &global_docs;
                let abort_ref = &abort;
                let count_progress_ref = &count_progress;
                handles.push(s.spawn(move || {
                    let res = count_worker(
                        idx,
                        source,
                        start,
                        end,
                        is_last_without_lf,
                        work_dir,
                        chunk_bytes,
                        unigram_budget,
                        bigram_budget,
                        global_docs_ref,
                        abort_ref,
                        count_progress_ref,
                    );
                    if res.is_err() {
                        abort_ref.store(true, Ordering::Relaxed);
                    }
                    res
                }));
            }
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or(Err(Status::IoError)))
                .collect()
        })
    };

    let mut token_total = 0u64;
    let mut document_count = 0u64;
    let mut unigram_by_part: Vec<Vec<FastHashMap<Box<[u8]>, u64>>> =
        (0..NUM_PARTITIONS).map(|_| Vec::with_capacity(active_workers)).collect();
    let mut bigram_by_part: Vec<Vec<FastHashMap<Box<[u8]>, u64>>> =
        (0..NUM_PARTITIONS).map(|_| Vec::with_capacity(active_workers)).collect();
    let mut unigram_disk_runs = Vec::new();
    let mut bigram_disk_runs = Vec::new();

    for res in count_results {
        let worker_out = res?;
        token_total += worker_out.tokens;
        document_count += worker_out.docs;
        for (p, map) in worker_out.unigram_mem.into_iter().enumerate() {
            if !map.is_empty() {
                unigram_by_part[p].push(map);
            }
        }
        for (p, map) in worker_out.bigram_mem.into_iter().enumerate() {
            if !map.is_empty() {
                bigram_by_part[p].push(map);
            }
        }
        unigram_disk_runs.extend(worker_out.unigram_disk);
        bigram_disk_runs.extend(worker_out.bigram_disk);
    }

    let merge_threads = active_workers.min(NUM_PARTITIONS);

    // Phase 2a: Parallel Unigram Partition Merge
    let mut freq_unigrams: Vec<FastHashMap<Box<[u8]>, u64>> =
        (0..NUM_PARTITIONS).map(|_| FastHashMap::default()).collect();
    let unique_unigrams_atomic = AtomicU64::new(0);

    if merge_threads <= 1 {
        for p in 0..NUM_PARTITIONS {
            let maps = std::mem::take(&mut unigram_by_part[p]);
            let mut local_unique = 0u64;
            let mut local_freq = FastHashMap::default();
            merge_partition(p, maps, &unigram_disk_runs, |tok, count| {
                local_unique += 1;
                if count >= min_count {
                    local_freq.insert(tok, count);
                }
                Ok(())
            })?;
            unique_unigrams_atomic.fetch_add(local_unique, Ordering::Relaxed);
            freq_unigrams[p] = local_freq;
        }
    } else {
        let part_input: Vec<Mutex<Vec<FastHashMap<Box<[u8]>, u64>>>> =
            unigram_by_part.into_iter().map(Mutex::new).collect();
        let part_output: Vec<Mutex<FastHashMap<Box<[u8]>, u64>>> =
            (0..NUM_PARTITIONS).map(|_| Mutex::new(FastHashMap::default())).collect();
        let next_p = AtomicUsize::new(0);
        let merge_err = AtomicBool::new(false);

        thread::scope(|s| {
            for _ in 0..merge_threads {
                let next_p_ref = &next_p;
                let part_in_ref = &part_input;
                let part_out_ref = &part_output;
                let disk_ref = &unigram_disk_runs;
                let unique_ref = &unique_unigrams_atomic;
                let err_ref = &merge_err;
                s.spawn(move || {
                    loop {
                        let p = next_p_ref.fetch_add(1, Ordering::Relaxed);
                        if p >= NUM_PARTITIONS || err_ref.load(Ordering::Relaxed) {
                            break;
                        }
                        let maps = std::mem::take(&mut *part_in_ref[p].lock().unwrap());
                        let mut local_unique = 0u64;
                        let mut local_freq = FastHashMap::default();
                        let res = merge_partition(p, maps, disk_ref, |tok, count| {
                            local_unique += 1;
                            if count >= min_count {
                                local_freq.insert(tok, count);
                            }
                            Ok(())
                        });
                        if res.is_err() {
                            err_ref.store(true, Ordering::Relaxed);
                            break;
                        }
                        unique_ref.fetch_add(local_unique, Ordering::Relaxed);
                        *part_out_ref[p].lock().unwrap() = local_freq;
                    }
                });
            }
        });
        if merge_err.load(Ordering::Relaxed) {
            return Err(Status::IoError);
        }
        freq_unigrams = part_output
            .into_iter()
            .map(|m| m.into_inner().unwrap())
            .collect();
    }

    for run in &unigram_disk_runs {
        let _ = fs::remove_file(&run.path);
    }

    // Phase 2b: Parallel Bigram Partition Merge & Scoring
    let mut eligible_partitions: Vec<FastHashSet<Box<[u8]>>> =
        (0..NUM_PARTITIONS).map(|_| FastHashSet::default()).collect();
    let unique_bigrams_atomic = AtomicU64::new(0);

    let score_bigram = |bigram_key: Box<[u8]>,
                        n_ab: u64,
                        local_eligible: &mut FastHashSet<Box<[u8]>>|
     -> Result<(), Status> {
        if n_ab >= min_count {
            let sep_pos = bigram_key
                .iter()
                .position(|&b| b == 0)
                .ok_or(Status::InvalidState)?;
            let a = &bigram_key[..sep_pos];
            let b = &bigram_key[sep_pos + 1..];
            let n_a = *freq_unigrams[partition_of(a)]
                .get(a)
                .ok_or(Status::InvalidState)?;
            let n_b = *freq_unigrams[partition_of(b)]
                .get(b)
                .ok_or(Status::InvalidState)?;
            let num = (n_ab - min_count) as u128 * token_total as u128;
            let den = n_a as u128 * n_b as u128;
            let score = (num as f64) / (den as f64);
            if score > threshold {
                local_eligible.insert(bigram_key);
            }
        }
        Ok(())
    };

    if merge_threads <= 1 {
        for p in 0..NUM_PARTITIONS {
            let maps = std::mem::take(&mut bigram_by_part[p]);
            let mut local_unique = 0u64;
            let mut local_eligible = FastHashSet::default();
            merge_partition(p, maps, &bigram_disk_runs, |bigram_key, n_ab| {
                local_unique += 1;
                score_bigram(bigram_key, n_ab, &mut local_eligible)
            })?;
            unique_bigrams_atomic.fetch_add(local_unique, Ordering::Relaxed);
            eligible_partitions[p] = local_eligible;
        }
    } else {
        let part_input: Vec<Mutex<Vec<FastHashMap<Box<[u8]>, u64>>>> =
            bigram_by_part.into_iter().map(Mutex::new).collect();
        let part_output: Vec<Mutex<FastHashSet<Box<[u8]>>>> =
            (0..NUM_PARTITIONS).map(|_| Mutex::new(FastHashSet::default())).collect();
        let next_p = AtomicUsize::new(0);
        let merge_err = AtomicBool::new(false);

        thread::scope(|s| {
            for _ in 0..merge_threads {
                let next_p_ref = &next_p;
                let part_in_ref = &part_input;
                let part_out_ref = &part_output;
                let disk_ref = &bigram_disk_runs;
                let unique_ref = &unique_bigrams_atomic;
                let err_ref = &merge_err;
                let score_fn = &score_bigram;
                s.spawn(move || {
                    loop {
                        let p = next_p_ref.fetch_add(1, Ordering::Relaxed);
                        if p >= NUM_PARTITIONS || err_ref.load(Ordering::Relaxed) {
                            break;
                        }
                        let maps = std::mem::take(&mut *part_in_ref[p].lock().unwrap());
                        let mut local_unique = 0u64;
                        let mut local_eligible = FastHashSet::default();
                        let res = merge_partition(p, maps, disk_ref, |bigram_key, n_ab| {
                            local_unique += 1;
                            score_fn(bigram_key, n_ab, &mut local_eligible)
                        });
                        if res.is_err() {
                            err_ref.store(true, Ordering::Relaxed);
                            break;
                        }
                        unique_ref.fetch_add(local_unique, Ordering::Relaxed);
                        *part_out_ref[p].lock().unwrap() = local_eligible;
                    }
                });
            }
        });
        if merge_err.load(Ordering::Relaxed) {
            return Err(Status::IoError);
        }
        eligible_partitions = part_output
            .into_iter()
            .map(|m| m.into_inner().unwrap())
            .collect();
    }

    for run in &bigram_disk_runs {
        let _ = fs::remove_file(&run.path);
    }
    drop(freq_unigrams);

    let unique_unigrams = unique_unigrams_atomic.load(Ordering::Relaxed);
    let unique_bigrams = unique_bigrams_atomic.load(Ordering::Relaxed);
    let has_any_eligible = eligible_partitions.iter().any(|s| !s.is_empty());

    // Phase 3: Parallel Corpus Rewriting
    let global_written_docs = AtomicU64::new(0);
    let write_progress_cb = |doc_num: u64| write_progress(doc_num, document_count);

    let joined_count = if active_workers == 1 {
        let (start, end) = shards[0];
        let out_file = File::create(destination).map_err(|_| Status::IoError)?;
        let buf_writer = BufWriter::with_capacity(IO_BUFFER_BYTES, out_file);
        let (joined, mut buf_writer) = write_worker(
            source,
            start,
            end,
            end == file_len && ends_without_lf,
            chunk_bytes,
            separator,
            &eligible_partitions,
            has_any_eligible,
            buf_writer,
            &global_written_docs,
            &abort,
            &write_progress_cb,
        )?;
        buf_writer.flush().map_err(|_| Status::IoError)?;
        Ok::<u64, Status>(joined)
    } else {
        let shard_paths: Vec<PathBuf> = (0..active_workers)
            .map(|idx| {
                if idx == 0 {
                    destination.to_path_buf()
                } else {
                    work_dir.join(format!("out-shard-{idx}.part"))
                }
            })
            .collect();

        let write_results: Vec<Result<u64, Status>> = thread::scope(|s| {
            let mut handles = Vec::with_capacity(active_workers);
            for (idx, &(start, end)) in shards.iter().enumerate() {
                let target_path = &shard_paths[idx];
                let is_last_without_lf = end == file_len && ends_without_lf;
                let eligible_ref = &eligible_partitions;
                let global_written_ref = &global_written_docs;
                let abort_ref = &abort;
                let write_progress_ref = &write_progress_cb;
                handles.push(s.spawn(move || {
                    let res = (|| {
                        let out_file = File::create(target_path).map_err(|_| Status::IoError)?;
                        let buf_writer = BufWriter::with_capacity(IO_BUFFER_BYTES, out_file);
                        let (joined, mut buf_writer) = write_worker(
                            source,
                            start,
                            end,
                            is_last_without_lf,
                            chunk_bytes,
                            separator,
                            eligible_ref,
                            has_any_eligible,
                            buf_writer,
                            global_written_ref,
                            abort_ref,
                            write_progress_ref,
                        )?;
                        buf_writer.flush().map_err(|_| Status::IoError)?;
                        Ok(joined)
                    })();
                    if res.is_err() {
                        abort_ref.store(true, Ordering::Relaxed);
                    }
                    res
                }));
            }
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or(Err(Status::IoError)))
                .collect()
        });

        let mut total_joined = 0u64;
        for res in write_results {
            total_joined += res?;
        }

        let dest_file = fs::OpenOptions::new()
            .append(true)
            .open(destination)
            .map_err(|_| Status::IoError)?;
        let mut dest_writer = BufWriter::with_capacity(IO_BUFFER_BYTES, dest_file);
        for part_path in shard_paths.iter().skip(1) {
            let mut part_file = File::open(part_path).map_err(|_| Status::IoError)?;
            std::io::copy(&mut part_file, &mut dest_writer).map_err(|_| Status::IoError)?;
            let _ = fs::remove_file(part_path);
        }
        dest_writer.flush().map_err(|_| Status::IoError)?;
        Ok::<u64, Status>(total_joined)
    }?;

    Ok(PhrasePassReport {
        input_tokens: token_total,
        documents: document_count,
        unique_unigrams,
        unique_bigrams,
        joined_count,
        output_tokens: token_total - joined_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_threaded_matches_single_threaded_and_chunk_boundaries() {
        let dir = std::env::temp_dir().join(format!("w2v-phrase-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let src = dir.join("input.txt");
        let mut data = Vec::new();
        for _ in 0..8000 {
            data.extend_from_slice(b"new york city new york\nalpha beta gamma\n\n");
        }
        data.extend_from_slice(b"  tail_without_newline  ");
        fs::write(&src, &data).unwrap();

        let out1 = dir.join("out1.txt");
        let out4 = dir.join("out4.txt");
        let work = dir.join("work");
        fs::create_dir_all(&work).unwrap();

        let r1 = run_phrase_pass(
            &src,
            &out1,
            &work,
            5,
            0.1,
            b"_",
            7,
            1,
            |_| Ok(()),
            |_, _| Ok(()),
        )
        .unwrap();

        let r4 = run_phrase_pass(
            &src,
            &out4,
            &work,
            5,
            0.1,
            b"_",
            64 * 1024,
            4,
            |_| Ok(()),
            |_, _| Ok(()),
        )
        .unwrap();

        assert_eq!(r1, r4);
        assert_eq!(fs::read(&out1).unwrap(), fs::read(&out4).unwrap());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn partitioned_spill_and_merge_matches_exact_counts() {
        let dir = std::env::temp_dir().join(format!("w2v-phrase-spill-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let mut parts1: Vec<FastHashMap<Box<[u8]>, u64>> =
            (0..NUM_PARTITIONS).map(|_| FastHashMap::default()).collect();
        let mut parts2: Vec<FastHashMap<Box<[u8]>, u64>> =
            (0..NUM_PARTITIONS).map(|_| FastHashMap::default()).collect();

        let k1 = b"alpha\0beta".as_slice();
        let k2 = b"beta\0gamma".as_slice();
        parts1[partition_of(k1)].insert(k1.into(), 4);
        parts2[partition_of(k1)].insert(k1.into(), 5);
        parts2[partition_of(k2)].insert(k2.into(), 2);

        let run1 = spill_partitions_to_disk(&mut parts1, dir.join("r1.prun")).unwrap();
        let run2 = spill_partitions_to_disk(&mut parts2, dir.join("r2.prun")).unwrap();

        let runs = vec![run1, run2];
        let mut merged2 = FastHashMap::default();
        for p in 0..NUM_PARTITIONS {
            merge_partition(p, Vec::new(), &runs, |k, c| {
                merged2.insert(k, c);
                Ok(())
            })
            .unwrap();
        }
        assert_eq!(merged2.get(k1), Some(&9));
        assert_eq!(merged2.get(k2), Some(&2));
        let _ = fs::remove_dir_all(&dir);
    }
}
