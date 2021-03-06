"""
Show how use this project
"""

# импорты
from pathlib import Path
from multiprocessing import Pool
from typing import Dict, Union, List
from os import system
import datetime
import yaml
import pandas as pd

from rapidfuzz import fuzz
import pytesseract

from search import FuzzySearcher, dummy_preprocess
from recognize import Recognizer

# для тессеракта
system("export TESSDATA_PREFIX='/usr/share/tesseract-ocr/4.00/tessdata'")
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# указываем пути
project_dir: Path = Path.cwd().parent
inp_dir: Path = project_dir / 'inp'
xlsx_dir: Path = inp_dir / 'xlsx'
scanned_pdf_dir: Path = inp_dir / 'scanned_pdf'
searchable_pdf_dir: Path = inp_dir / 'searchable_pdf'
out_dir: Path = project_dir / 'out' / str(datetime.datetime.now()).split('.', maxsplit=1)[0]
log_path: Path = out_dir / 'log.txt'
output_path: Path = out_dir / 'output.xlsx'

out_dir.mkdir(parents=True, exist_ok=True)

with open(inp_dir / "config.yaml", "r", encoding='utf-8') as f:
    config: Dict[str, Union[bool, int]] = yaml.safe_load(f)
with open(log_path, 'a', encoding='utf-8') as file:
    print(config, file=file, flush=True)

with open(inp_dir / 'keywords.txt', encoding='utf-8') as f:
    keywords_not_preprocessed: List[str] = [line.replace('\n', ' ') for line in f.readlines()]
    keywords_not_preprocessed = list(filter(lambda x: x not in (' ', ''), keywords_not_preprocessed))

# define FuzzySearcher and Recognizer
fuzzy: FuzzySearcher = FuzzySearcher(ratio=fuzz.token_sort_ratio,
                                     # функция, определяющая близость фраз с значениями в (0, 100)
                                     partial_ratio=fuzz.partial_ratio,
                                     # как ratio, но для функции search_in_pdf_fast
                                     conf_threshold_percent=80,
                                     # порог уверенности, если ratio выше погрога, то фраза считается найденной
                                     preprocess=dummy_preprocess,
                                     # функция для препроцессинга
                                     keywords=keywords_not_preprocessed,
                                     # список с ключевыми словами
                                     log_path=project_dir / 'log.txt')
# путь для файла с логами

recognizer: Recognizer = Recognizer(dpi=600,
                                    # dots per inch, 300-600 рекомендуется
                                    log_path=project_dir / 'log.txt',
                                    # путь для файла с логами
                                    lang='ru',
                                    # язык из доступных для тессеракта
                                    searchable_pdf_dir=project_dir / 'inp' / 'searchable_pdf',
                                    # путь до папки с текстовыми pdf
                                    preprocess_config={'resize': False,             # какие преобразования из
                                                       'adaptiveThreshold': False,  # Recognizer.image_preprocess
                                                       'bilateralFilter': False})   # применять к изображениям


# multiprocessing recognize and search
if __name__ == '__main__':
    with Pool(processes=4) as pool:
        # pool.map(recognizer.scanned2searchable, scanned_pdf_dir.glob('*.pdf'))

        result_xlsx: pd.DataFrame = fuzzy.try_concat_result(pool.map(fuzzy.search_in_xlsx,
                                                                     xlsx_dir.glob('*.xlsx')))
        result_pdf: pd.DataFrame = fuzzy.try_concat_result(pool.map(fuzzy.search_in_pdf,
                                                                    searchable_pdf_dir.glob('*.pdf')))

print(f'\n{"* " * 35}\n')
print('result xlsx')
print(result_xlsx)

print(f'\n{"* " * 35}\n')
print('result pdf')
print(result_pdf)
