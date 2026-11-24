use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

/// Fast BPE merge step — given a word as Vec<String> and a merge pair,
/// return a new Vec<String> with all occurrences of (a, b) merged to "ab".
fn apply_merge(word: &[String], a: &str, b: &str) -> Vec<String> {
    let merged = format!("{}{}", a, b);
    let mut out = Vec::with_capacity(word.len());
    let mut i = 0;
    while i < word.len() {
        if i + 1 < word.len() && word[i] == a && word[i + 1] == b {
            out.push(merged.clone());
            i += 2;
        } else {
            out.push(word[i].clone());
            i += 1;
        }
    }
    out
}

/// Count bigram pair frequencies across all words.
fn count_pairs(vocab: &HashMap<Vec<String>, u64>) -> HashMap<(String, String), u64> {
    let mut counts: HashMap<(String, String), u64> = HashMap::new();
    for (word, freq) in vocab.iter() {
        for i in 0..word.len().saturating_sub(1) {
            *counts
                .entry((word[i].clone(), word[i + 1].clone()))
                .or_insert(0) += freq;
        }
    }
    counts
}

#[pyclass]
struct RustBPE {
    vocab_size: usize,
    lowercase: bool,
    merges: Vec<(String, String)>,
    token_to_id: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    merge_index: HashMap<(String, String), usize>,
    encode_cache: HashMap<String, Vec<u32>>,
}

#[pymethods]
impl RustBPE {
    #[new]
    #[pyo3(signature = (vocab_size=50257, lowercase=false))]
    fn new(vocab_size: usize, lowercase: bool) -> Self {
        RustBPE {
            vocab_size,
            lowercase,
            merges: Vec::new(),
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            merge_index: HashMap::new(),
            encode_cache: HashMap::new(),
        }
    }

    /// Train BPE on a list of strings. Learns merges bottom-up.
    fn train(&mut self, texts: Vec<String>) -> PyResult<()> {
        // Collect word frequencies
        let mut word_freqs: HashMap<String, u64> = HashMap::new();
        for text in &texts {
            let t = if self.lowercase { text.to_lowercase() } else { text.clone() };
            for word in t.split_whitespace() {
                let key = word.chars().map(|c| c.to_string()).collect::<Vec<_>>().join(" ")
                    + " </w>";
                *word_freqs.entry(key).or_insert(0) += 1;
            }
        }

        // Build vocab: each key is a space-separated symbol sequence
        let mut vocab: HashMap<Vec<String>, u64> = word_freqs
            .iter()
            .map(|(k, &v)| (k.split(' ').map(|s| s.to_string()).collect(), v))
            .collect();

        // Collect base symbols
        let mut base_syms: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for word in vocab.keys() {
            for sym in word {
                base_syms.insert(sym.clone());
            }
        }

        self.token_to_id = base_syms
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        self.token_to_id.insert("[UNK]".to_string(), self.token_to_id.len() as u32);
        self.id_to_token = self.token_to_id.iter().map(|(k, &v)| (v, k.clone())).collect();
        self.merges.clear();
        self.merge_index.clear();

        let num_merges = self.vocab_size.saturating_sub(self.token_to_id.len());

        for _ in 0..num_merges {
            let pair_counts = count_pairs(&vocab);
            if pair_counts.is_empty() {
                break;
            }
            // Pick the most frequent pair (ties broken lexicographically)
            let best = pair_counts
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
                .unwrap();
            let (a, b) = best.0.clone();
            let merged = format!("{}{}", a, b);

            // Apply merge across vocab
            let new_vocab: HashMap<Vec<String>, u64> = vocab
                .into_iter()
                .map(|(word, freq)| (apply_merge(&word, &a, &b), freq))
                .collect();
            vocab = new_vocab;

            // Register new token
            let new_id = self.token_to_id.len() as u32;
            self.token_to_id.entry(merged.clone()).or_insert(new_id);
            self.id_to_token.entry(new_id).or_insert(merged);
            self.merge_index.insert((a.clone(), b.clone()), self.merges.len());
            self.merges.push((a, b));
        }

        self.encode_cache.clear();
        Ok(())
    }

    /// Encode a string to token ids.
    fn encode(&mut self, text: String) -> PyResult<Vec<u32>> {
        if let Some(ids) = self.encode_cache.get(&text) {
            return Ok(ids.clone());
        }

        let t = if self.lowercase { text.to_lowercase() } else { text.clone() };
        let unk_id = *self.token_to_id.get("[UNK]").unwrap_or(&0);

        let mut all_ids: Vec<u32> = Vec::new();
        for word in t.split_whitespace() {
            let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            symbols.push("</w>".to_string());

            // Apply all merges in order using merge_index for O(1) lookup
            loop {
                let mut best_rank = usize::MAX;
                let mut best_pos = None;
                for i in 0..symbols.len().saturating_sub(1) {
                    let pair = (symbols[i].clone(), symbols[i + 1].clone());
                    if let Some(&rank) = self.merge_index.get(&pair) {
                        if rank < best_rank {
                            best_rank = rank;
                            best_pos = Some(i);
                        }
                    }
                }
                match best_pos {
                    None => break,
                    Some(pos) => {
                        let merged = format!("{}{}", symbols[pos], symbols[pos + 1]);
                        symbols[pos] = merged;
                        symbols.remove(pos + 1);
                    }
                }
            }

            for sym in &symbols {
                all_ids.push(*self.token_to_id.get(sym).unwrap_or(&unk_id));
            }
        }

        self.encode_cache.insert(text, all_ids.clone());
        Ok(all_ids)
    }

    /// Decode token ids to a string.
    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        let mut out = String::new();
        for id in ids {
            if let Some(tok) = self.id_to_token.get(&id) {
                if tok == "</w>" {
                    out.push(' ');
                } else if tok.ends_with("</w>") {
                    out.push_str(&tok[..tok.len() - 4]);
                    out.push(' ');
                } else {
                    out.push_str(tok);
                }
            }
        }
        Ok(out.trim_end().to_string())
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    #[getter]
    fn merges(&self, py: Python<'_>) -> PyResult<PyObject> {
        let list = PyList::empty(py);
        for (a, b) in &self.merges {
            list.append((a.clone(), b.clone()))?;
        }
        Ok(list.into())
    }

    /// Load merges and vocab from dicts (for warm-start / save-load).
    fn load_state(&mut self, token_to_id: HashMap<String, u32>,
                  merges: Vec<(String, String)>) -> PyResult<()> {
        self.token_to_id = token_to_id;
        self.id_to_token = self.token_to_id.iter().map(|(k, &v)| (v, k.clone())).collect();
        self.merge_index = merges
            .iter()
            .enumerate()
            .map(|(i, (a, b))| ((a.clone(), b.clone()), i))
            .collect();
        self.merges = merges;
        self.encode_cache.clear();
        Ok(())
    }

    fn get_vocab(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = PyDict::new(py);
        for (k, v) in &self.token_to_id {
            d.set_item(k, v)?;
        }
        Ok(d.into())
    }
}

#[pymodule]
fn vp_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustBPE>()?;
    Ok(())
}
