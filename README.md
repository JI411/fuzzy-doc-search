# fuzzy-doc-search: Проект для распознавания (OCR) pdf документов, поиска ключевых фраз в xlsx и текстовых pdf.

Для решения задачи OCR используется класс Recognizer (из recognize.py). Для поиске - FuzzySearcher (из search.py).
Для запуска из конфига используетс файл fuzzy_doc_search.py, конфиг лежит в папке inp_path, в example_config.yaml есть пояснения к каждому параметру в конфиге.
____
Для использования OCR необходимо указать путь до тессеракта в 20-21 строчках fuzzy_doc_search.py. Для удобного встраивания в другие модули все методы снабжены описаниями.

Что можно спокойно менять:
- Другая система для OCR
- Функция препроцессинга для текста и ключевых фраз (передаётся в FuzzySearcher)
- Запуск не из конфига, а как удобно
____

Использование (весь код есть в src/quick_start.py):

```python


# импорты
from pathlib import Path
from multiprocessing import Pool
from os import system
import datetime
import yaml
import pandas as pd

from rapidfuzz import fuzz
import pytesseract

from search import FuzzySearcher, dummy_preprocess
from recognize import Recognizer


# указываем пути
system("export TESSDATA_PREFIX='/usr/share/tesseract-ocr/4.00/tessdata'")
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
project_dir = Path.cwd().parent
inp_dir = project_dir / 'inp'
xlsx_dir = inp_dir / 'xlsx'
scanned_pdf_dir = inp_dir / 'scanned pdf'
searchable_pdf_dir = inp_dir / 'searchable pdf'
out_dir = project_dir / 'out' / str(datetime.datetime.now())
log_path = out_dir / 'log.txt'
output_path = out_dir / 'output.xlsx'

# define FuzzySearcher and Recognizer
fuzzy: FuzzySearcher = FuzzySearcher(ratio=fuzz.token_sort_ratio,        # функция, определяющая близость фраз с значениями в (0, 100)
                                     partial_ratio=fuzz.partial_ratio,   # как ratio, но для функции search_in_pdf_fast
                                     conf_threshold_percent=80,          # порог уверенности, если ratio выше погрога, то фраза считается найденной
                                     preprocess=dummy_preprocess,        # функция для препроцессинга
                                     keywords=keywords_not_preprocessed, # список с ключевыми словами
                                     log_path=project_dir / 'log.txt')   # путь для файла с логами

recognizer: Recognizer = Recognizer(dpi=600,  # dots per inch, численная мера качества фото при распознавании, 300-600 рекомендуется
                                    log_path=project_dir / 'log.txt',  # путь для файла с логами
                                    lang='ru',                         # язык из доступных для тессеракта
                                    searchable_pdf_dir=project_dir / 'inp' / 'searchable pdf',  # путь до папки с текстовыми pdf
                                    preprocess_config={'resize': False, 'adaptiveThreshold': False, 'bilateralFilter': False})
                                    # какие преобразования над изображениями из Recognizer.image_preprocess применять

# multiprocessing recognize and search
with Pool(processes=4) as pool:
    pool.map(recognizer.scanned2searchable, scanned_pdf_dir.glob('*.pdf'))

    result_xlsx: pd.DataFrame = fuzzy.try_concat_result(pool.map(fuzzy.search_in_xlsx,
                                                                 xlsx_dir.glob('*.xlsx')))
    result_pdf: pd.DataFrame = fuzzy.try_concat_result(pool.map(fuzzy.search_in_pdf,
                                                                searchable_pdf_dir.glob('*.pdf')))
    result_pdf_fast: pd.DataFrame = fuzzy.try_concat_result(pool.map(fuzzy.search_in_pdf_fast,
                                                                     searchable_pdf_dir.glob('*.pdf')))
                                                                     
```

