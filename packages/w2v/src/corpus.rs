use crate::config::{MAX_TOKEN_LENGTH, Status};
use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

/// Immutable corpus path and byte length, as in `reference/src/corpus/corpus.c`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Corpus {
    pub path: PathBuf,
    pub byte_size: usize,
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
        })
    }

    pub fn tokenizer(&self, byte_offset: usize) -> Result<Tokenizer<BufReader<File>>, Status> {
        let mut file = File::open(&self.path).map_err(|_| Status::IoError)?;
        file.seek(SeekFrom::Start(byte_offset as u64))
            .map_err(|_| Status::IoError)?;
        Ok(Tokenizer::new(BufReader::new(file)))
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
}
impl<R: Read> Tokenizer<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            pending_newline: false,
        }
    }

    pub fn read_token(&mut self) -> Result<TokenRead, Status> {
        let mut token = [0u8; MAX_TOKEN_LENGTH];
        let mut length = 0;
        let mut at_eof = false;
        loop {
            let character = if self.pending_newline {
                self.pending_newline = false;
                Some(b'\n')
            } else {
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
                _ if length < MAX_TOKEN_LENGTH - 1 => {
                    token[length] = character;
                    length += 1;
                }
                _ => {}
            }
        }
        let visible_length = token[..length]
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(length);
        Ok(TokenRead {
            token: token[..visible_length].to_vec(),
            at_eof,
        })
    }
}
