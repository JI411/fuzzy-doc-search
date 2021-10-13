"""
Start script from user config from inp_path
"""
# pylint: disable=line-too-long

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

# setup tesseract configuration
system("export TESSDATA_PREFIX='/usr/share/tesseract-ocr/4.00/tessdata'")
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

project_dir: Path = Path.cwd().parent
inp_dir: Path = project_dir / 'inp'
xlsx_dir: Path = inp_dir / 'xlsx'
scanned_pdf_dir: Path = inp_dir / 'scanned_pdf'
searchable_pdf_dir: Path = inp_dir / 'searchable_pdf'
out_dir: Path = project_dir / 'out' / str(datetime.datetime.now())
log_path: Path = out_dir / 'log.txt'
output_path: Path = out_dir / 'output.xlsx'


out_dir.mkdir(parents=True, exist_ok=True)

with open(inp_dir / "config.yaml", "r", encoding='utf-8') as f:
    config: Dict[str, Union[bool, int]] = yaml.safe_load(f)
with open(out_dir / "config.yaml", "w", encoding='utf-8') as f:
    yaml.dump(config, f)

print(config)
with open(log_path, 'a', encoding='utf-8') as file:
    print(config, file=file, flush=True)

if __name__ == '__main__':

    if config.get('recognize', False):
        preprocess_image: bool = config.get('preprocess_image', False)
        recognizer: Recognizer = Recognizer(dpi=config.get('dpi', 300), log_path=log_path,
                                            searchable_pdf_dir=searchable_pdf_dir,
                                            lang=config.get('lang', 'ru'),
                                            preprocess_config={'resize': preprocess_image,
                                                               'adaptiveThreshold': preprocess_image,
                                                               'bilateralFilter': preprocess_image})
        with Pool(processes=4) as pool:
            pool.map(recognizer.scanned2searchable, scanned_pdf_dir.glob('*.pdf'))

    if config.get('search', False):
        with open(inp_dir / 'keywords.txt', encoding='utf-8') as f:
            keywords_not_preprocessed: List[str] = [line.replace('\n', ' ') for line in f.readlines()]
            keywords_not_preprocessed = list(filter(lambda x: x not in (' ', ''), keywords_not_preprocessed))

        writer: pd.ExcelWriter = pd.ExcelWriter(output_path)  # pylint: disable=abstract-class-instantiated

        fuzzy: FuzzySearcher = FuzzySearcher(ratio=fuzz.ratio if config['word_order'] else fuzz.token_sort_ratio,
                                             partial_ratio=fuzz.partial_ratio,
                                             conf_threshold_percent=config.get('conf_threshold_percent', 80),
                                             preprocess=dummy_preprocess,
                                             keywords=keywords_not_preprocessed,
                                             log_path=log_path)
        with Pool(processes=4) as pool:
            result_xlsx: pd.DataFrame = fuzzy.try_concat_result(pool.map(fuzzy.search_in_xlsx,
                                                                         xlsx_dir.glob('*.xlsx')))
            result_xlsx.to_excel(writer, 'xlsx', index=False)

            if config.get('fast_search_in_pdf', False):
                result_pdf_fast: pd.DataFrame = fuzzy.try_concat_result(pool.map(fuzzy.search_in_pdf_fast,
                                                                                 searchable_pdf_dir.glob('*.pdf')))
                result_pdf_fast.to_excel(writer, 'pdf_fast', index=False)

            else:
                result_pdf: pd.DataFrame = fuzzy.try_concat_result(pool.map(fuzzy.search_in_pdf,
                                                                            searchable_pdf_dir.glob('*.pdf')))
                result_pdf.to_excel(writer, 'pdf', index=False)

        writer.save()
