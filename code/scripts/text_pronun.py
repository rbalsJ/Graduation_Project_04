import numpy as np
import re
from db_manager import DatabaseManager
from keras.models import load_model
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, TimeDistributed, Activation, RepeatVector, Embedding, LSTM, Masking, Dropout


# 경로
ENG_DB_PATH = 'data/processed/database/eng_database.db'
IPA_DB_PATH = 'data/processed/database/ipa_database.db'
ALPHABET_DB_PATH = 'data/processed/database/alphabet_database.db'
MODEL_PATH = 'data/our_sam_model.h5'
INPUT_SOURCE_PATH  = 'data/raw/test_sentence/eng_source.txt'
OUTPUT_SOURCE_PATH  = 'data/processed/test_results/eng_source_lstm.txt'


class Seq2SeqModel:
    SPACE = [' ']
    PAD = ['<pad>']
    ENGLISH_LETTERS = PAD + [chr(i) for i in range(ord('A'), ord('Z')+1)] + [chr(i) for i in range(ord('a'), ord('z')+1)] + SPACE
    IDX2ENGLISH = dict(enumerate(ENGLISH_LETTERS))
    ENGLISH2IDX = {v: k for k, v in IDX2ENGLISH.items()}
    
    CONSONANTS = [chr(letter) for letter in range(ord('ㄱ'), ord('ㅎ')+1)]
    VOWEL = [chr(letter) for letter in range(ord('ㅏ'), ord('ㅣ')+1)]
    KOREAN_LETTERS = PAD + CONSONANTS + VOWEL + SPACE
    IDX2KOREAN = dict(enumerate(KOREAN_LETTERS))
    KOREAN2IDX = {v: k for k, v in IDX2KOREAN.items()}

    def __init__(self):
        self.model = load_model('data/our_sam_model.h5')
        self.max_len_src = 50
        self.max_len_trg = 50

    def eng_to_kor_translation(self, str):
        predic = self.model.predict([[self.ENGLISH2IDX.get(char, 0) for char in str] + [0] * (self.max_len_src - len(str))])
        _idx2korean = [np.argmax(val) for val in predic[0]]
        return [self.IDX2KOREAN[idx] for idx in _idx2korean if idx != 0]

    @staticmethod
    # 자음, 모음으로 이루어진 한글 합치기
    def combine_korean_characters(characters):
        # 초성, 중성, 종성 리스트
        cho = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        jung = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
        jong = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

        # 모음과 모음 사이에 하나의 자음만 있는 경우 처리
        idx = 0
        while idx < len(characters) - 2:
            if characters[idx] in jung and characters[idx+1] in cho and characters[idx+2] in jung:
                characters.insert(idx+1, '')
                idx += 2
            else:
                idx += 1

        result = []
        idx = 0
        while idx < len(characters):
            c = characters[idx]
            # c가 초성인 경우
            if c in cho:
                # 현재 문자와 다음 두 문자를 조합하여 한글 문자 생성
                try:
                    a = cho.index(c)
                    b = jung.index(characters[idx+1])
                    c = jong.index(characters[idx+2])
                    combined_char = chr(a * 21 * 28 + b * 28 + c + 0xAC00)
                    result.append(combined_char)
                    idx += 3
                except (IndexError, ValueError):  # 다음 문자가 없거나 종성이 없는 경우
                    try:
                        a = cho.index(c)
                        b = jung.index(characters[idx+1])
                        combined_char = chr(a * 21 * 28 + b * 28 + 0xAC00)
                        result.append(combined_char)
                        idx += 2
                    except (IndexError, ValueError):  # 다음 문자가 없는 경우
                        result.append(c)
                        idx += 1
            else:
                result.append(c)
                idx += 1

        return ''.join(result)


class TextPronunciation:
    def __init__(self):
        self.db_manager_eng = DatabaseManager(ENG_DB_PATH)
        self.db_manager_ipa = DatabaseManager(IPA_DB_PATH)
        self.db_manager_alp = DatabaseManager(ALPHABET_DB_PATH)
        self.seq2seq_model = Seq2SeqModel()  # Assuming seq2seq class is defined elsewhere in your code
        
        self.total_count = 0
        self.single_alphabet_count = 0
        self.our_sam_database_count = 0
        self.ipa_database_count = 0
        self.upper_case_count = 0
        self.no_result_count = 0

    def get_pronunciation(self, word, db_manager):
        char_list_pronun = [db_manager.check_word_existence(char)[5] for char in word]
        return ''.join(char_list_pronun)

    def word_pronunciation(self, word):
        word_str = str(word)
        self.total_count += 1  # Increment the word count each time this method is called

        #Single alphabet case
        if len(word_str) == 1 and word_str.isalpha() and word_str.isascii():
            self.single_alphabet_count += 1
            return self.get_pronunciation(word, self.db_manager_alp)

        # Check in 우리샘사전 DB
        elif self.db_manager_eng.check_word_existence(word):
            # print("우리샘사전 DB\n")
            self.our_sam_database_count += 1
            return self.db_manager_eng.check_word_existence(word)[5]

        # Check in IPA DB
        elif self.db_manager_ipa.check_word_existence(word):
            # print("IPA DB\n")
            self.ipa_database_count += 1
            return self.db_manager_ipa.check_word_existence(word)[5]
        # else:
        #     return word_str

        # All uppercase case
        elif word_str.isupper():
            # print("알파벳이고 모두 대문자 DB\n")
            self.upper_case_count += 1
            return self.get_pronunciation(word, self.db_manager_alp)

        # 딥러닝 case
        else:
            # print("Deep러닝 DB\n")
            self.no_result_count += 1
            print(word)
            result = self.seq2seq_model.eng_to_kor_translation(word)  # 인자 수정
            return self.seq2seq_model.combine_korean_characters(result)
        
        # # 딥러닝 case
        # # print("Deep러닝 DB\n")
        # self.no_result_count += 1
        # print(word)
        # result = self.seq2seq_model.eng_to_kor_translation(word)  # 인자 수정
        # return self.seq2seq_model.combine_korean_characters(result)
        
    def sentence_pronunciation(self, txt):
        pattern = re.compile(r'[a-zA-Z]+')
        sentences = pattern.findall(txt)
        for value in sentences:
            txt = txt.replace(value, self.word_pronunciation(value), 1)
        return txt
    


def main():
    text_pronunciation = TextPronunciation()
    # transformed_sentence = text_pronunciation.word_pronunciation("specialtly")
    # print(transformed_sentence)

    with open(INPUT_SOURCE_PATH, 'r', encoding='utf-8') as infile:
        sentences = infile.readlines()

    output_sentences = []
    for sentence in sentences:
        transformed_sentence = text_pronunciation.sentence_pronunciation(sentence.strip())
        output_sentences.append(transformed_sentence)

    with open(OUTPUT_SOURCE_PATH, 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(output_sentences))

    print(f"Processing complete. Results saved to {OUTPUT_SOURCE_PATH}")
    print(f"Total words processed: {text_pronunciation.total_count}")
    print(f"single_alphabet_count: {text_pronunciation.single_alphabet_count}")
    print(f"our_sam_database_count: {text_pronunciation.our_sam_database_count}")
    print(f"ipa_database_count: {text_pronunciation.ipa_database_count}")
    print(f"upper_case_count: {text_pronunciation.upper_case_count}")
    print(f"deep_learning_case_count: {text_pronunciation.no_result_count}")
    print(f"Processing Rate: {1 - (text_pronunciation.no_result_count / text_pronunciation.total_count)}")

if __name__ == "__main__":
    main()