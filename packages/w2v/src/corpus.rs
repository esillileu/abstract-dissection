use crate::config::{MAX_TOKEN_LENGTH, Status};
use std::{
    fs::File,
    io::{BufRead, BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

/// Immutable corpus path and byte length, as in `reference/src/corpus/corpus.c`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Corpus {
    pub path: PathBuf,
    pub byte_size: usize,
    /// Preserve full byte tokens for phrase corpora; false keeps C-compatible truncation.
    pub full_tokens: bool,
}
impl Corpus {
    pub fn create(path: impl AsRef<Path>) -> Result<Self, Status> {
        let path = path.as_ref();
        if path.as_os_str().is_empty() {
            return Err(Status::InvalidArgument);
        }
        let mut file = File::open(path).map_err(|_| Status::IoError)?;
        let length = file.seek(SeekFrom::End(0)).map_err(|_| Status::IoError)?;
        if length > isize::MAX as u64 {
            return Err(Status::IoError);
        }
        Ok(Self {
            path: path.to_path_buf(),
            byte_size: length as usize,
            full_tokens: false,
        })
    }

    pub fn tokenizer(&self, byte_offset: usize) -> Result<Tokenizer<BufReader<File>>, Status> {
        let mut file = File::open(&self.path).map_err(|_| Status::IoError)?;
        file.seek(SeekFrom::Start(byte_offset as u64))
            .map_err(|_| Status::IoError)?;
        let mut tokenizer = Tokenizer::new(BufReader::new(file));
        tokenizer.full_tokens = self.full_tokens;
        Ok(tokenizer)
    }

    /// Contiguous byte shards aligned forward to LF; adjacent workers share a boundary.
    pub fn sentence_shard(&self, worker: usize, workers: usize) -> Result<(usize, usize), Status> {
        let align = |offset: usize| -> Result<usize, Status> {
            if offset == 0 || offset >= self.byte_size {
                return Ok(offset.min(self.byte_size));
            }
            let mut file = File::open(&self.path).map_err(|_| Status::IoError)?;
            file.seek(SeekFrom::Start((offset - 1) as u64))
                .map_err(|_| Status::IoError)?;
            let mut byte = [0u8];
            file.read_exact(&mut byte).map_err(|_| Status::IoError)?;
            if byte[0] == b'\n' {
                return Ok(offset);
            }
            let mut reader = BufReader::new(file);
            let mut position = offset;
            loop {
                let buffer = reader.fill_buf().map_err(|_| Status::IoError)?;
                if buffer.is_empty() {
                    return Ok(position);
                }
                let count = buffer
                    .iter()
                    .position(|&b| b == b'\n')
                    .map(|p| p + 1)
                    .unwrap_or(buffer.len());
                let ended = buffer[count - 1] == b'\n';
                position += count;
                reader.consume(count);
                if ended {
                    return Ok(position);
                }
            }
        };
        Ok((
            align(self.byte_size / workers * worker)?,
            if worker + 1 == workers {
                self.byte_size
            } else {
                align(self.byte_size / workers * (worker + 1))?
            },
        ))
    }

    pub fn sentence_tokenizer(
        &self,
        start: usize,
        end: usize,
    ) -> Result<Tokenizer<BufReader<File>>, Status> {
        let mut tokenizer = self.tokenizer(start)?;
        tokenizer.remaining_bytes = Some(end - start);
        Ok(tokenizer)
    }

    pub fn digest(&self) -> Result<String, Status> {
        let mut file = File::open(&self.path).map_err(|_| Status::IoError)?;
        let mut hash = crate::identity::StableDigest::new();
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let count = file.read(&mut buffer).map_err(|_| Status::IoError)?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
        }
        Ok(hash.finish())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenRead {
    pub token: Vec<u8>,
    pub at_eof: bool,
}

/// Byte tokenizer with the C `ungetc('\n')` behavior.
pub struct Tokenizer<R: Read> {
    reader: R,
    pending_newline: bool,
    pub(crate) token_byte_limit: usize,
    full_tokens: bool,
    remaining_bytes: Option<usize>,
}
impl<R: Read> Tokenizer<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            pending_newline: false,
            token_byte_limit: 0,
            full_tokens: false,
            remaining_bytes: None,
        }
    }

    pub fn read_token(&mut self) -> Result<TokenRead, Status> {
        loop {
            let mut token = Vec::with_capacity(MAX_TOKEN_LENGTH);
            let mut length = 0;
            let mut at_eof = false;
            loop {
                let character = if self.pending_newline {
                    self.pending_newline = false;
                    Some(b'\n')
                } else if self.remaining_bytes == Some(0) {
                    None
                } else {
                    if let Some(remaining) = &mut self.remaining_bytes {
                        *remaining -= 1;
                    }
                    let mut byte = [0u8; 1];
                    match self.reader.read(&mut byte) {
                        Ok(0) => None,
                        Ok(_) => Some(byte[0]),
                        Err(_) => return Err(Status::IoError),
                    }
                };
                let Some(character) = character else {
                    at_eof = true;
                    break;
                };
                match character {
                    b'\r' => continue,
                    b'\n' if length != 0 => {
                        self.pending_newline = true;
                        break;
                    }
                    b'\n' => {
                        return Ok(TokenRead {
                            token: b"</s>".to_vec(),
                            at_eof: false,
                        });
                    }
                    b' ' | b'\t' if length != 0 => break,
                    b' ' | b'\t' => continue,
                    _ if self.full_tokens || length < MAX_TOKEN_LENGTH - 1 => {
                        if self.token_byte_limit > 0 && length >= self.token_byte_limit {
                            return Err(Status::InvalidArgument);
                        }
                        token.try_reserve(1).map_err(|_| Status::OutOfMemory)?;
                        token.push(character);
                        length += 1;
                    }
                    _ => {}
                }
            }
            let visible_length = if self.full_tokens {
                length
            } else {
                token.iter().position(|&byte| byte == 0).unwrap_or(length)
            };
            if visible_length == 0 && !at_eof {
                continue;
            }
            let at_eof = at_eof && !(self.full_tokens && visible_length > 0);
            return Ok(TokenRead {
                token: {
                    token.truncate(visible_length);
                    token
                },
                at_eof,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn full_tokens_are_distinct_while_default_retains_c_truncation() {
        let prefix = vec![b'a'; 120];
        let mut input = prefix.clone();
        input.extend_from_slice(b"x ");
        input.extend_from_slice(&prefix);
        input.extend_from_slice(b"y\n");
        let mut legacy = Tokenizer::new(Cursor::new(input.clone()));
        assert_eq!(legacy.read_token().unwrap().token, vec![b'a'; 99]);
        assert_eq!(legacy.read_token().unwrap().token, vec![b'a'; 99]);
        let mut exact = Tokenizer::new(Cursor::new(input));
        exact.full_tokens = true;
        let first = exact.read_token().unwrap();
        let second = exact.read_token().unwrap();
        assert_eq!(first.token.len(), 121);
        assert_eq!(second.token.len(), 121);
        assert_ne!(first.token, second.token);
        assert_eq!(exact.read_token().unwrap().token, b"</s>");
        assert!(exact.read_token().unwrap().at_eof);
    }

    #[test]
    fn full_tokens_emit_the_last_token_before_eof() {
        let mut exact = Tokenizer::new(Cursor::new(vec![b'x'; 200]));
        exact.full_tokens = true;
        let token = exact.read_token().unwrap();
        assert_eq!(token.token.len(), 200);
        assert!(!token.at_eof);
        assert!(exact.read_token().unwrap().at_eof);
    }
}
