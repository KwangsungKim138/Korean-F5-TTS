# ruff: noqa: F722 F821

from __future__ import annotations

import os
import random
from collections import defaultdict
from importlib.resources import files

import rjieba
import torch
from pypinyin import Style, lazy_pinyin
from torch.nn.utils.rnn import pad_sequence


# seed everything


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# helpers


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def is_package_available(package_name: str) -> bool:
    try:
        import importlib

        package_exists = importlib.util.find_spec(package_name) is not None
        return package_exists
    except Exception:
        return False


# tensor helpers


def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


def maybe_masked_mean(t: float["b n d"], mask: bool["b n"] = None) -> float["b d"]:
    if not exists(mask):
        return t.mean(dim=1)

    t = torch.where(mask[:, :, None], t, torch.tensor(0.0, device=t.device))
    num = t.sum(dim=1)
    den = mask.float().sum(dim=1)

    return num / den.clamp(min=1.0)


# simple utf-8 tokenizer, since paper went character based
def list_str_to_tensor(text: list[str], padding_value=-1) -> int["b nt"]:
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]  # ByT5 style
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text


# char tokenizer, based on custom dataset's extracted .txt file
def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
) -> int["b nt"]:
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text


# Get tokenizer


def get_tokenizer(dataset_name, tokenizer: str = "pinyin"):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "kor_grapheme" for Korean Jamo decomposition (no G2P), need .txt vocab_file
                - "kor_allophone" for Korean G2A conversion, need .txt vocab_file
                - "kor_phoneme" for standard phoneme tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "kor_grapheme", derived from Korean Jamos
                - if use "kor_allophone", derived from Korean allophones
                - if use "kor_phoneme", derived from phonemes
                - if use "byte", set to 256 (unicode byte range)
    """
    if tokenizer in ["pinyin", "char", "kor_grapheme", "kor_allophone", "kor_phoneme"]:
        tokenizer_path = os.path.join(files("f5_tts").joinpath("../../data"), f"{dataset_name}_{tokenizer}/vocab.txt")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256

    elif tokenizer == "custom":
        with open(dataset_name, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size


# Korean G2A constants and helpers
try:
    from g2pk2 import G2p
except ImportError:
    G2p = None

_g2p_instance = None


def get_g2p():
    global _g2p_instance
    if _g2p_instance is None:
        if G2p is None:
            raise ImportError("g2pk2 is not installed. Please install it with `pip install g2pk2`.")
        # Suppress g2pk2/mecab init messages (e.g. "mecab installed") on first use
        import contextlib

        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                _g2p_instance = G2p()
    return _g2p_instance

# 1. 초/중/종성 자소 자모 집합 정의
GRAPHEME_CHOSEONG = ["ㄱ","ㄲ","ㄴ","ㄷ","ㄸ","ㄹ","ㅁ","ㅂ","ㅃ","ㅅ", "ㅆ","ㅇ","ㅈ","ㅉ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]
GRAPHEME_JUNGSEONG = ["ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ","ㅗ","ㅘ", "ㅙ","ㅚ","ㅛ","ㅜ","ㅝ","ㅞ","ㅟ","ㅠ","ㅡ","ㅢ","ㅣ"]
GRAPHEME_JONGSEONG = ["","ㄱ","ㄲ","ㄳ","ㄴ","ㄵ","ㄶ","ㄷ","ㄹ","ㄺ", "ㄻ","ㄼ","ㄽ","ㄾ","ㄿ","ㅀ","ㅁ","ㅂ","ㅄ","ㅅ", "ㅆ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]

# 2. 한국어 음소 자모 집합 정의
PHONEME_CONSONANTS = ["ㄱ","ㄲ","ㄴ","ㄷ","ㄸ","ㄹ","ㅁ","ㅂ","ㅃ","ㅅ", "ㅆ","ㅇ","ㅈ","ㅉ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]
PHONEME_VOWELS = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ', 'ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ', 'ㅣ']

# 3. Target phonemes - 변이음 세부 분류 규칙을 구성하는 음소들
PHONEMES_I = ["ㄱ", "ㄷ", "ㅂ", "ㅈ"] # 어두에서 무성음화되는 평음
PHONEMES_P = ["ㄴ", "ㄹ", "ㅅ", "ㅆ", "ㅎ"] # [j], [i] 앞에서 변이하는 자음
PHONEMES_C = ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅇ"] # 종성 phonemes
VOWELS_Y = ["ㅣ", "ㅑ", "ㅕ", "ㅛ", "ㅠ", "ㅖ", "ㅒ", "ㅟ"] # [j], [i] 계열 모음

# 4. allophone 기호
MARK_INIT = "ⁱ"  # 어두 초성 (Word-initial -> Voiceless)
MARK_CODA = "ᶜ"  # 종성 (Unreleased/Lateral 등)
MARK_PAL  = "ʲ"  # 구개음화 (Palatalized)

def _text_to_pronunciation(text: str) -> str:
    """
    텍스트 -> 발음열 변환
    """
    g2p = get_g2p()
    return g2p(text)

def _pronunciation_to_eojeols(text: str) -> list[str]:
    """
    발음열 -> 어절 단위 분해
    """
    return text.split(' ')

def _syllable_to_phonemes(syllable: str) -> list[str]:
    """
    음절(syllable) -> 음소(phoneme) 분해
    """
    if ord("가") <= ord(syllable) <= ord("힣"):
        base = ord(syllable) - ord("가")
        cho = GRAPHEME_CHOSEONG[base // 588]
        jung = GRAPHEME_JUNGSEONG[(base % 588) // 28]
        jong = GRAPHEME_JONGSEONG[base % 28]
        return [cho, jung, jong]
    else:
        return [syllable]

def _classify_into_allophones(
    phonemes: list[str],
    is_eojeol_initial: bool,
    ) -> list[str]:
    """
    음소열(sequence of phonemes)을 변이음(allophone) 단위로 분류,
    sequence of allophones 출력
    """
    allophones = []

    if len(phonemes) <= 2:
        return phonemes
    else:
        cho, jung, jong = phonemes[:3]

    # 1) 초성(onset)
    if is_eojeol_initial and cho in PHONEMES_I:
        allophones.append(cho + MARK_INIT)  # 예: ㄱ -> ㄱⁱ
    elif cho in PHONEMES_P and jung in VOWELS_Y:
        allophones.append(cho + MARK_PAL)   # 예: ㅅ -> ㅅʲ
    else: # allophone 기호가 없는 초성으로, 음소에 단일 음가(phone)가 대응됨을 의미
        allophones.append(cho)

    # 2) 중성(nucleus)
    allophones.append(jung)

    # 3) 종성(coda)
    if jong:
        allophones.append(jong + MARK_CODA)

    return allophones

def convert_char_to_allophone(text_list: list[str]) -> list[list[str]]:
    """
    Convert text list to Korean allophone list
    """
    final_text_list = []
    
    for text in text_list:
        result = []
        # 1. 문자열을 발음열로 변환하고
        pronunciation = _text_to_pronunciation(text)

        # 2. 공백을 기준으로 어절 분리
        eojeols = _pronunciation_to_eojeols(pronunciation)

        # 3. 각 어절에 대해
        for eojeol in eojeols:
            # 4. 음절,
            for i, syllable in enumerate(eojeol):
                # 5. 음소 단위로 분해한 뒤
                phonemes = _syllable_to_phonemes(syllable)
                # 6. 변이음 분류
                allophones = _classify_into_allophones(phonemes,
                                          is_eojeol_initial = (i == 0))
                result.extend(allophones)
            result.append(' ')
        
        # remove last space
        if result and result[-1] == ' ':
            result.pop()
            
        final_text_list.append(result)

    return final_text_list

def convert_char_to_grapheme(text_list: list[str]) -> list[list[str]]:
    """
    Convert text list to Korean Grapheme (Jamo) list (No G2P)
    """
    final_text_list = []
    for text in text_list:
        result = []
        for char in text:
            if char == ' ':
                result.append(' ')
            else:
                # Decompose syllable
                jamos = _syllable_to_phonemes(char)
                # Filter out empty jongseong if any
                result.extend([j for j in jamos if j])
        
        final_text_list.append(result)
    return final_text_list

def convert_char_to_phoneme(text_list: list[str]) -> list[list[str]]:
    """
    Convert text list to Korean Standard Phoneme list (G2P applied, No Allophone markers)
    """
    final_text_list = []
    for text in text_list:
        result = []
        # 1. G2P
        pronunciation = _text_to_pronunciation(text)
        # 2. Eojeol split
        eojeols = _pronunciation_to_eojeols(pronunciation)
        
        for eojeol in eojeols:
            for syllable in eojeol:
                # 3. Decompose
                phonemes = _syllable_to_phonemes(syllable)
                result.extend([p for p in phonemes if p])
            result.append(' ')
            
        if result and result[-1] == ' ':
            result.pop()
            
        final_text_list.append(result)
    return final_text_list


# convert char to pinyin


def convert_char_to_pinyin(text_list, polyphone=True):
    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return (
            "\u3100" <= c <= "\u9fff"  # common chinese characters
        )

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in rjieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list


# filter func for dirty data with many repetitions


def repetition_found(text, length=2, tolerance=10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] += 1
    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
    return False


# get the empirically pruned step for sampling


def get_epss_timesteps(n, device, dtype):
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return dt * torch.tensor(t, device=device, dtype=dtype)
