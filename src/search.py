"""
Class for fuzzy string search in searchable pdf, xlsx
"""
# pylint: disable=line-too-long

from pathlib import Path
from multiprocessing import Pool

from math import ceil, floor
from typing import List, Callable, Union, Generator, Dict
import datetime

import regex as re
import pandas as pd
import fitz
# import docx2txt

from rapidfuzz import fuzz


# import swifter
# print(swifter.__version__)


def dummy_ratio(first_text: str, second_text: str) -> float:
    """
    check string equality
    """
    return first_text == second_text


def dummy_preprocess(text: str) -> str:
    """
    replace \n, \t, needless spaces and symbols
    """
    text = re.sub(r"[^А-Яа-я0-9()\s,]", " ", text)
    text = text.replace('\t', ' ').replace('\n', ' ').strip().lower()
    # text = re.sub(r"[^A-Za-z0-9()\s,]", " ", text)
    return text


class FuzzySearcher:
    """
    Basic class for a search in docs
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, ratio: Callable[[str, str], float], partial_ratio: Callable[[str, str], float],
                 conf_threshold_percent: Union[int, float], log_path: Path,
                 preprocess: Callable[[str], str], keywords: List[str]) -> None:
        """
        :param ratio: ratio func, return ratio of two string in range 0-100
        :param partial_ratio: like ratio, but can be used to find partial ratio
        :param conf_threshold_percent:  match if ratio > conf, must be in range 0-100
        :param preprocess: function to preprocess texts from docs and keywords
        :param keywords: keywords to find in docs
        """
        # pylint: disable=too-many-arguments

        self.ratio = ratio
        self.partial_ratio = partial_ratio
        self.log_path = log_path
        self.preprocess = preprocess
        self.keywords_original = tuple(keywords)
        self.keywords = tuple(filter(lambda x: x != '', (preprocess(i) for i in keywords)))
        if not self.keywords:
            self.log("ValueError 'Empty keywords list after preprocessing'")
            raise ValueError('Empty keywords list after preprocessing')
        elif len(self.keywords) != len(self.keywords_original):
            self.log("ValueError 'Some keywords remove after preprocessing'")
            self.log(f'keywords after preprocessing: {self.keywords}')
            raise ValueError('Some keywords remove after preprocessing')

        if 0 < conf_threshold_percent < 100:
            self.conf_t: float = float(conf_threshold_percent)
        else:
            raise ValueError('conf_threshold_percent must be between 0 and 100')

        len_keywords = [len(keyword) for keyword in self.keywords]
        self.min_len_matched_text: int = max(1, floor(min(len_keywords) * self.conf_t / 100))
        self.max_len_matched_text: int = ceil(max(len_keywords) * 100 / self.conf_t)

        self.log(f'FuzzySearcher initialization: {str(datetime.datetime.now())}')
        self.log(f'Keywords: {self.keywords}')

    @staticmethod
    def try_concat_result(gen: Union[List[pd.DataFrame],
                                     Generator[pd.DataFrame, None, None]]) -> pd.DataFrame:
        """
        :param gen: generator with dataframe from another methods
        :return: one dataframe with all results or ValueError describe
        """
        try:
            result: pd.DataFrame = pd.concat((g for g in gen if g is not None), ignore_index=True)
        except ValueError as error:
            result: pd.DataFrame = pd.DataFrame([f'ValueError: {error}', 'check path to directory'])
        return result

    def log(self, *args, **kwargs) -> None:
        """
        Write log to self.log file
        """
        print(*args, **kwargs)
        with open(self.log_path, 'a', encoding='utf-8') as file:
            print(*args, **kwargs, file=file, flush=True)

    def search_in_xlsx(self, xlsx_path: Path) -> Union[pd.DataFrame, None]:
        """
        Use self.ratio to find keywords in file

        :param xlsx_path: path to xlsx file
        :return: dataframe with columns:
                 keyword original, keyword,  document name, sheet name, string #, context
        """
        result_from_one_xlsx: List[pd.DataFrame] = []
        for sheet_name, dataframe in pd.read_excel(xlsx_path, sheet_name=None, header=None).items():
            if dataframe.empty:
                continue
            dataframe: pd.DataFrame = dataframe.fillna('').astype(str)
            for i, keyword in enumerate(self.keywords):
                save_if_match: Callable[[str], str] = lambda x, k=keyword: \
                    x if self.ratio(k, self.preprocess(x)) > self.conf_t else ''
                # sub_df: pd.DataFrame = dataframe.swifter.progress_bar(False).allow_dask_on_strings(True)
                # sub_df = sub_df.applymap(save_if_match)
                sub_df: pd.DataFrame = dataframe.applymap(save_if_match)
                sub_df = sub_df.loc[sub_df.iloc[:, 0] == ''].drop_duplicates(keep=False)
                sub_df = sub_df.apply(pd.Series.unique, axis=1)
                sub_df = sub_df.apply(lambda unique_matches: [match for match in unique_matches if match])
                sub_df = pd.DataFrame({'keyword': keyword,
                                       'keyword original': self.keywords_original[i],
                                       'string #': sub_df.index + 1,
                                       'context': sub_df})
                sub_df['sheet name'] = sheet_name
                sub_df['document name'] = xlsx_path.name
                sub_df = sub_df[['keyword original', 'keyword', 'document name', 'sheet name', 'string #', 'context']]
                result_from_one_xlsx.append(sub_df)
        if result_from_one_xlsx:
            self.log(f'Done xlsx search: {xlsx_path.name}')
            return pd.concat(result_from_one_xlsx)
        return None

    def search_in_pdf(self, pdf_path: Path) -> Union[pd.DataFrame, None]:
        """
        Use self.ratio to find keywords in file

        :param pdf_path: path to searchable pdf file
        :return: dataframe with columns: keyword original, keyword,  document name, page number, context
        """
        result_from_one_pdf: List[Dict] = []
        # noinspection PyUnresolvedReferences
        with fitz.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf):
                result_from_one_page: Dict[str, List[pd.Interval]] = {}
                # keys == keywords, values == keys position in text

                text: str = page.getText('text')
                if not text:
                    continue

                text = self.preprocess(text)
                len_text = len(text)
                for len_chunk in range(self.min_len_matched_text, self.max_len_matched_text + 1):
                    for start, end in ((i, i + len_chunk) for i in range(0, len_text, len_chunk)):
                        chunk = text[start: end]
                        for keyword in self.keywords:
                            if self.ratio(keyword, chunk) > self.conf_t:

                                new_interval = pd.Interval(left=start, right=end)  # closed='both'
                                if keyword not in result_from_one_page.keys():
                                    result_from_one_page[keyword] = [new_interval]
                                else:
                                    add_new_interval = True
                                    for i, interval in enumerate(result_from_one_page[keyword]):
                                        if interval.overlaps(new_interval):
                                            result_from_one_page[keyword][i] = pd.Interval(
                                                min(interval.left, new_interval.left),
                                                max(interval.right, new_interval.right))
                                            add_new_interval = False
                                    if add_new_interval:
                                        result_from_one_page[keyword].append(new_interval)

                for keyword_original, keyword in zip(self.keywords_original, self.keywords):
                    if keyword in result_from_one_page.keys():
                        for interval in result_from_one_page[keyword]:
                            result_from_one_pdf.append({'keyword original': keyword_original,
                                                        'keyword': keyword,
                                                        'document name': pdf_path.name,
                                                        'page number': page_num,
                                                        'context': text[max(0, interval.left - 10):
                                                                        min(len_text, interval.right + 10)]})
        if result_from_one_pdf:
            self.log(f'Done pdf search: {pdf_path.name}')
            return pd.DataFrame(result_from_one_pdf).sort_values(by=['keyword original', 'keyword', 'page number'])
        return None

    def search_in_pdf_fast(self, pdf_path: Path) -> Union[pd.DataFrame, None]:
        """
        Use self.partial_ratio to find keywords in file
        Attention: don't return context, return one or zero dataframe row for every page

        :param pdf_path: path to searchable pdf file
        :return: dataframe with columns: keyword original, keyword,  document name, page num
        """
        result_from_one_pdf: List[Dict] = []
        # noinspection PyUnresolvedReferences
        with fitz.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf):
                text: str = page.getText('text')
                if not text:
                    continue

                text = self.preprocess(text)
                for i, keyword in enumerate(self.keywords):
                    if self.partial_ratio(keyword, text) > self.conf_t:
                        result_from_one_pdf.append({'keyword original': self.keywords_original[i],
                                                    'keyword': keyword + ' ',
                                                    'document name': pdf_path.name,
                                                    'page number': page_num})
        if result_from_one_pdf:
            self.log(f'Done fast pdf search: {pdf_path.name}')
            return pd.DataFrame(result_from_one_pdf).sort_values(by=['keyword original',
                                                                     'page number']).drop_duplicates()
        return None


if __name__ == '__main__':
    project_dir = Path.cwd().parent
    inp_dir = project_dir / 'inp'
    xlsx_dir = inp_dir / 'xlsx'
    searchable_pdf_dir = inp_dir / 'searchable pdf'

    with open(inp_dir / 'keywords.txt', encoding='utf-8') as f:
        keywords_not_preprocessed = [line.replace('\n', ' ') for line in f.readlines()]
        keywords_not_preprocessed = [line for line in keywords_not_preprocessed if line not in (' ', '')]

    fuzzy: FuzzySearcher = FuzzySearcher(ratio=fuzz.token_sort_ratio,  # ratio=fuzz.ratio
                                         partial_ratio=fuzz.partial_ratio,
                                         conf_threshold_percent=80,
                                         preprocess=dummy_preprocess,
                                         keywords=keywords_not_preprocessed,
                                         log_path=project_dir / 'log.txt')

    # result_xlsx: pd.DataFrame = fuzzy.try_concat_result((fuzzy.search_in_xlsx(xlsx_path)
    #                                                      for xlsx_path in xlsx_dir.glob('*.xlsx')))
    #
    # result_pdf: pd.DataFrame = fuzzy.try_concat_result((fuzzy.search_in_pdf(pdf_path)
    #                                                     for pdf_path in searchable_pdf_dir.glob('*.pdf')))
    #
    # result_pdf_fast: pd.DataFrame = fuzzy.try_concat_result((fuzzy.search_in_pdf_fast(pdf_path)
    #                                                          for pdf_path in searchable_pdf_dir.glob('*.pdf')))

    with Pool(processes=4) as pool:
        result_xlsx: pd.DataFrame = fuzzy.try_concat_result(pool.map(fuzzy.search_in_xlsx,
                                                                     xlsx_dir.glob('*.xlsx')))
        result_pdf: pd.DataFrame = fuzzy.try_concat_result(pool.map(fuzzy.search_in_pdf,
                                                                    searchable_pdf_dir.glob('*.pdf')))
        result_pdf_fast: pd.DataFrame = fuzzy.try_concat_result(pool.map(fuzzy.search_in_pdf_fast,
                                                                         searchable_pdf_dir.glob('*.pdf')))

    print(f'\n{"* " * 35}\n')
    print(result_xlsx)
    print(f'\n{"* " * 35}\n')
    print(result_pdf)
    print(f'\n{"* " * 35}\n')
    print(result_pdf_fast)
