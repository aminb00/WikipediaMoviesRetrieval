from __future__ import annotations
import re
import unicodedata
from typing import Iterable, List, Optional, Set
import logging

try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS
except Exception:  # spaCy yoksa da çalışalım
    spacy = None
    SPACY_STOP_WORDS = set()

log = logging.getLogger(__name__)


class Tokenizer:
    """
    A friendly, batteries-included text tokenizer 💬

    Goals:
    - Practical defaults (lowercase, diacritics-strip, lemma)
    - Respect common “specials” (URLs, emails, #hashtags, @mentions)
    - Numbers: keep 4-digit years; bucket the rest as <NUM>
    - Optional stopword removal, min token length
    - Works even if `en_core_web_sm` yok: zarif bir fallback ile.

    Quick start:
        tk = Tokenizer(keep_urls=True, remove_stopwords=True)
        tokens = tk("Data mining in 2021: see https://example.com #nlp @you")

    Notes:
    - spaCy model bulunamazsa: basit bir whitespace/punct splitter + naive “lemma=lower”.
    - `__call__` -> `tokenize` kısayolu.
    """

    # --- Regexler (tek yerde, test etmesi kolay) ---
    _URL_RE     = re.compile(r"(?i)\bhttps?://\S+")
    _EMAIL_RE   = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")
    _HASHTAG_RE = re.compile(r"(?i)(?<!\w)#\w+")
    _MENTION_RE = re.compile(r"(?i)(?<!\w)@\w+")
    _YEAR_RE    = re.compile(r"^(19\d{2}|20\d{2})$")
    _NUM_RE     = re.compile(r"^\d+([.,]\d+)?$")
    _CLEAN_RE   = re.compile(r"[^a-z0-9<>]+")  # sadeleştirme için

    # normalize: fancy quotes/dashes to ascii
    _QUOTES_DASHES = str.maketrans({
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "–": "-", "—": "-", "-": "-", "−": "-", "·": ".",
    })

    def __init__(
        self,
        keep_urls: bool = True,
        keep_emails: bool = True,
        keep_hashtags: bool = True,
        keep_mentions: bool = True,
        remove_stopwords: bool = False,
        custom_stopwords: Optional[Iterable[str]] = None,
        min_token_len: int = 1,
        model: str = "en_core_web_sm",
        disable_spacy: bool = False,
    ) -> None:
        self.keep_urls = keep_urls
        self.keep_emails = keep_emails
        self.keep_hashtags = keep_hashtags
        self.keep_mentions = keep_mentions
        self.remove_stopwords = remove_stopwords
        self.min_token_len = max(0, int(min_token_len))

        # stopword set
        self.stopwords: Set[str] = set(SPACY_STOP_WORDS)
        if custom_stopwords:
            self.stopwords |= {w.lower() for w in custom_stopwords}

        # spaCy yükle (yoksa kibarca fallback)
        self._use_spacy = False
        self._nlp = None
        if not disable_spacy and spacy is not None:
            try:
                self._nlp = spacy.load(model, disable=["ner", "parser", "senter", "textcat"])
                self._use_spacy = True
            except Exception as e:
                log.warning("spaCy model '%s' yüklenemedi (%s). Basit fallback kullanılacak.", model, e)
        else:
            if disable_spacy:
                log.info("spaCy devre dışı. Basit fallback kullanılacak.")
            else:
                log.info("spaCy bulunamadı. Basit fallback kullanılacak.")

    # ---- küçük kalite dokunuşları ----
    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)

    def __repr__(self) -> str:
        return (
            f"Tokenizer(spacy={'on' if self._use_spacy else 'off'}, "
            f"urls={self.keep_urls}, emails={self.keep_emails}, "
            f"hashtags={self.keep_hashtags}, mentions={self.keep_mentions}, "
            f"stopwords={'on' if self.remove_stopwords else 'off'}, "
            f"min_len={self.min_token_len})"
        )

    # ---- yardımcılar ----
    @staticmethod
    def _strip_diacritics(text: str) -> str:
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

    def _normalize_surface(self, text: str) -> str:
        # lowercase + diacritics + quote/dash normalize
        text = (text or "").translate(self._QUOTES_DASHES).lower()
        return self._strip_diacritics(text)

    def _preextract(self, text: str) -> Set[str]:
        specials: Set[str] = set()
        if self.keep_urls:
            specials.update(m.group(0) for m in self._URL_RE.finditer(text))
        if self.keep_emails:
            specials.update(m.group(0) for m in self._EMAIL_RE.finditer(text))
        if self.keep_hashtags:
            specials.update(m.group(0) for m in self._HASHTAG_RE.finditer(text))
        if self.keep_mentions:
            specials.update(m.group(0) for m in self._MENTION_RE.finditer(text))
        return specials

    def _normalize_number(self, tok: str) -> str:
        if self._YEAR_RE.match(tok):
            return tok
        return "<NUM>" if self._NUM_RE.match(tok) else tok

    def _should_keep(self, tok: str) -> bool:
        if not tok:
            return False
        if len(tok) < self.min_token_len:
            return False
        if self.remove_stopwords and tok in self.stopwords:
            return False
        return True

    # ---- public api ----
    def tokenize(self, text: str) -> List[str]:
        if not text or not isinstance(text, str):
            return []

        text_norm = self._normalize_surface(text)
        specials = self._preextract(text_norm)

        if self._use_spacy:
            doc = self._nlp(text_norm)
            out: List[str] = []
            for t in doc:
                if t.is_space or t.is_punct:
                    continue

                raw = t.text
                if raw in specials:
                    # özel belirteci olduğu gibi koru
                    tok = self._CLEAN_RE.sub("", raw)
                    if self._should_keep(tok):
                        out.append(tok)
                    continue

                lemma = t.lemma_ if t.lemma_ != "-PRON-" else t.text
                tok = self._normalize_number(lemma)
                tok = self._CLEAN_RE.sub("", tok)
                if self._should_keep(tok):
                    out.append(tok)
            return out

        # ---- Fallback yolu: spaCy yoksa basit ama tutarlı bir ayrıştırma ----
        # URL/e-mail/hashtag/mention'ları önce *yerine koy*, sonra split et.
        placeholders = {}
        def _stash(match):
            key = f"<SPL>{len(placeholders)}"
            placeholders[key] = match.group(0)
            return f" {key} "

        tmp = text_norm
        # sırayla yerleştir (URL → email → hashtag → mention)
        tmp = self._URL_RE.sub(_stash, tmp) if self.keep_urls else tmp
        tmp = self._EMAIL_RE.sub(_stash, tmp) if self.keep_emails else tmp
        tmp = self._HASHTAG_RE.sub(_stash, tmp) if self.keep_hashtags else tmp
        tmp = self._MENTION_RE.sub(_stash, tmp) if self.keep_mentions else tmp

        rough = re.split(r"\s+|[^\w<>]+", tmp)
        out: List[str] = []
        for r in rough:
            if not r:
                continue
            if r.startswith("<SPL>") and r in placeholders:
                tok = self._CLEAN_RE.sub("", placeholders[r])
            else:
                tok = self._normalize_number(r)
                tok = self._CLEAN_RE.sub("", tok)
            if self._should_keep(tok):
                out.append(tok)
        return out

    def tokenize_batch(self, texts: List[str], batch_size: int = 64) -> List[List[str]]:
        """
        Batch tokenization (spaCy varsa pipe, yoksa fallback döngü).
        """
        normed = [self._normalize_surface(t) if isinstance(t, str) else "" for t in texts]

        if self._use_spacy:
            results: List[List[str]] = []
            for doc in self._nlp.pipe(normed, batch_size=batch_size):
                specials = self._preextract(doc.text)
                tokens: List[str] = []
                for t in doc:
                    if t.is_space or t.is_punct:
                        continue
                    raw = t.text
                    if raw in specials:
                        tok = self._CLEAN_RE.sub("", raw)
                    else:
                        lemma = t.lemma_ if t.lemma_ != "-PRON-" else t.text
                        tok = self._normalize_number(lemma)
                        tok = self._CLEAN_RE.sub("", tok)
                    if self._should_keep(tok):
                        tokens.append(tok)
                results.append(tokens)
            return results

        # fallback toplu
        return [self.tokenize(t) for t in normed]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tk = Tokenizer(remove_stopwords=True, min_token_len=2)
    s = "Friends, Romans — email me at j.black@mail.yahoo.com on 03/12/1991; price is 12.50€! #NLP @You https://ex.am/ple"
    print(repr(tk))
    print(tk.tokenize(s))
