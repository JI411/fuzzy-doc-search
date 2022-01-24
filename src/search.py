"""
Class for fuzzy string search in searchable_pdf, xlsx
"""
# pylint: disable=line-too-long

from pathlib import Path
from multiprocessing import Pool

from math import ceil, floor
from typing import Callable, Union, Generator, Collection, List, Dict
import datetime

import regex as re
import pandas as pd
import fitz
from numpy import nan
from rapidfuzz import fuzz


def dummy_ratio(first_text: str, second_text: str) -> float:
    """
    check string equality
    """
    return first_text == second_text


def dummy_preprocess(text: str) -> str:
    """
    replace \n, \t, needless spaces and symbols
    """
    text = text.lower().replace('ё', 'е')
    text = re.sub(r"[^a-zа-я0-9()\s,]", "", text)
    text = text.replace('\t', ' ').replace('\n', ' ').strip()
    return text


class FuzzySearcher:
    """
    Basic class for a search in docs
    """

    def __init__(self,
                 conf_threshold_percent: Union[int, float],
                 log_path: Path,
                 keywords: List[str],
                 preprocess: Callable[[str], str],
                 ratio: Callable[[str, str], float] = None,
                 partial_ratio: Callable[[str, str], float] = None) -> None:
        """
        :param ratio: ratio func, return ratio of two string in range 0-100
        :param partial_ratio: like ratio, but can be used to find partial ratio
        :param conf_threshold_percent:  match if ratio > conf, must be integer in range 1-100
        :param preprocess: function to preprocess texts from docs and keywords
        :param keywords: keywords to find in docs
        """
        # pylint: disable=too-many-arguments
        # because it is necessary to define all params in __init__
        # we can contain some in config, but this variant more clear
        self.ratio = ratio
        self.partial_ratio = partial_ratio
        self.log_path = log_path
        self.preprocess = preprocess
        self.keywords: Dict[str, Collection[str]] = {
            'original': tuple(keywords),
            'processed': tuple(filter(lambda x: x != '', (preprocess(i) for i in keywords)))
        }
        self.conf_t: int = int(conf_threshold_percent)

        self.check_keywords()
        self.check_conf_t()

        len_keywords = [len(keyword) for keyword in self.keywords['processed']]
        self.len_keywords: Dict[str, int] = {'min': max(1, floor(min(len_keywords) * self.conf_t / 100)),
                                             'max': ceil(max(len_keywords) * 100 / self.conf_t)}

        self.log(f'FuzzySearcher initialization: {datetime.datetime.now()}')
        self.log(f"Keywords: {self.keywords['processed']}")

    def check_keywords(self) -> None:
        """Check list of keywords non-empty and we don't lost any words in preprocessing """
        if not self.keywords['processed']:
            self.log("ValueError 'Empty keywords list after preprocessing'")
            raise ValueError('Empty keywords list after preprocessing')
        if len(self.keywords['processed']) != len(self.keywords['original']):
            self.log("ValueError 'Some keywords remove after preprocessing'")
            self.log(f"keywords after preprocessing: {self.keywords['processed']}")
            raise ValueError('Some keywords remove after preprocessing')

    def check_conf_t(self) -> None:
        """Check conf_threshold_percent in range 0-100"""
        if self.conf_t < 0 or self.conf_t >= 100:
            raise ValueError('conf_threshold_percent must be float between 0 and 100')

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
        Write log to self.log file and print in console
        """
        print(*args, **kwargs)
        with open(self.log_path, 'a', encoding='utf-8') as file:
            print(*args, **kwargs, file=file, flush=True)

    def search_in_xlsx(self, xlsx_path: Path) -> Union[pd.DataFrame, None]:
        """
        Use self.partial_ratio to find keywords in file

        :param xlsx_path: path to xlsx file
        :return: dataframe with columns:
                 keyword original, keyword,  document name, sheet name, string #, context
        """
        if self.partial_ratio is None:
            self.log("Define partial_ratio function to find in xlsx")
            raise Exception("Define partial_ratio function to find in xlsx")
        result_from_one_xlsx: List[pd.DataFrame] = []
        for sheet_name, dataframe in pd.read_excel(xlsx_path, sheet_name=None, header=None).items():
            if dataframe.empty:
                continue
            dataframe = dataframe.fillna('').astype(str)
            for keyword_original, keyword in zip(self.keywords["original"], self.keywords["processed"]):
                if self.conf_t == 100:
                    save_if_match: Callable[[str], str] = lambda x, k=keyword: \
                        x if k in self.preprocess(x) else nan
                else:
                    save_if_match: Callable[[str], str] = lambda x, k=keyword: \
                        x if self.partial_ratio(k, self.preprocess(x)) > self.conf_t else nan

                sub_df: pd.DataFrame = dataframe.applymap(save_if_match)
                sub_df = sub_df.dropna(how='all').astype(str)
                if sub_df.empty:
                    continue
                sub_df = sub_df.apply(lambda row: row.str.strip().to_numpy().flatten(), axis=1)
                sub_df = sub_df.map(lambda row_as_array: [x for x in row_as_array if x != 'nan'])

                sub_df = pd.DataFrame({'keyword': keyword,
                                       'keyword original': keyword_original,
                                       'string #': sub_df.index + 1,
                                       'context': sub_df})
                sub_df['sheet name'] = sheet_name
                sub_df['document name'] = xlsx_path.name
                sub_df = sub_df.explode('context').fillna('')
                sub_df = sub_df[sub_df['context'] != '']
                result_from_one_xlsx.append(sub_df)

        if result_from_one_xlsx:
            self.log(f'Done xlsx search: {xlsx_path.name}')
            columns_order = ['keyword original', 'keyword', 'document name', 'sheet name', 'string #', 'context']
            return pd.concat(result_from_one_xlsx)[columns_order]
        self.log(f'Nothing find in {xlsx_path.name}')
        return None

    @staticmethod
    def update_result_from_one_page_pdf(new_interval: pd.Interval,
                                        result_from_one_page: Dict[str, List[pd.Interval]], keyword: str):
        """
        Add new_interval to list of chunks with keyword if interval not interception with another
        Else filter list of chunks - combine it.
        Example of combine: [(1, 5), (2, 4), (3, 7), (7, 10)] ---> [(1, 7), (7, 10)]

        :param new_interval: pd.Interval - chunk position in text
        :param result_from_one_page: keys == keywords, values == list of keys position in text
        :param keyword:  new keyword to add
        :return: updated result_from_one_page
        """
        if keyword in result_from_one_page:
            add_new_interval = True
            for i, interval in enumerate(result_from_one_page[keyword]):
                if interval.overlaps(new_interval):
                    result_from_one_page[keyword][i] = pd.Interval(
                        min(interval.left, new_interval.left),
                        max(interval.right, new_interval.right))
                    add_new_interval = False
            if add_new_interval:
                result_from_one_page[keyword].append(new_interval)
        else:
            result_from_one_page[keyword] = [new_interval]
        return result_from_one_page

    def result_from_one_page_2_one_pdf(self, pdf_path: Path, page_num: int, text: str,
                                       result_from_one_page: Dict[str, List[pd.Interval]],
                                       result_from_one_pdf: List[Dict]):
        """
        Update result_from_one_pdf in search_in_pdf method.

        :param pdf_path: path to pdf document containing page, that we are processing
        :param page_num: pdf page parameter, start from zero
        :param text: text from page, that we are processing
        :param result_from_one_page: dict that containing keys == keywords, values == list of keys position in text
        :param result_from_one_pdf: list with dict result_from_one_page for every page in pdf
        :return: updated result_from_one_pdf
        """
        # pylint: disable=too-many-arguments
        # because it is necessary to write together the page number, path to pdf, context, etc.
        len_text = len(text)
        for keyword_original, keyword in zip(self.keywords['original'], self.keywords['processed']):
            if keyword in result_from_one_page:
                for interval in result_from_one_page[keyword]:
                    result_from_one_pdf.append({'keyword original': keyword_original,
                                                'keyword': keyword,
                                                'document name': pdf_path.name,
                                                'page number': page_num + 1,
                                                'context': text[max(0, interval.left - 10):
                                                                min(len_text, interval.right + 10)]})
        return result_from_one_pdf

    def search_in_pdf(self, pdf_path: Path) -> Union[pd.DataFrame, None]:
        """
        Use self.ratio to find keywords in file

        Split text from every page to chunks, add chunk to result if ratio(chunk, keyword) > conf_t,
        filter list of chunks - if we have nonempty interception of intervals (start and end exclude), we combine it.
        Example of combine: [(1, 5), (2, 4), (3, 7), (7, 10)] ---> [(1, 7), (7, 10)]

        :param pdf_path: path to searchable_pdf file
        :return: dataframe with columns: keyword original, keyword,  document name, page number, context
        """
        if self.ratio is None:
            self.log("Define ratio function to find in pdf")
            raise Exception("Define ratio function to find in pdf")
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
                for len_chunk in range(self.len_keywords['min'], self.len_keywords['max'] + 1):
                    for start, end in ((i, i + len_chunk) for i in range(0, len_text, len_chunk)):
                        chunk = text[start: end]
                        for keyword in self.keywords['processed']:
                            if self.ratio(keyword, chunk) > self.conf_t:

                                result_from_one_page = self.update_result_from_one_page_pdf(
                                    new_interval=pd.Interval(left=start, right=end),
                                    result_from_one_page=result_from_one_page,
                                    keyword=keyword)

                result_from_one_pdf = self.result_from_one_page_2_one_pdf(pdf_path=pdf_path,
                                                                          page_num=page_num,
                                                                          text=text,
                                                                          result_from_one_page=result_from_one_page,
                                                                          result_from_one_pdf=result_from_one_pdf)

        if result_from_one_pdf:
            self.log(f'Done pdf search: {pdf_path.name}')
            return pd.DataFrame(result_from_one_pdf).sort_values(by=['keyword original', 'keyword', 'page number'])
        self.log(f'Nothing find in {pdf_path.name}')
        return None


if __name__ == '__main__':
    project_dir = Path.cwd().parent
    inp_dir = project_dir / 'inp'
    xlsx_dir = inp_dir / 'xlsx'
    searchable_pdf_dir = inp_dir / 'searchable_pdf'

    with open(inp_dir / 'keywords.txt', encoding='utf-8') as f:
        keywords_not_preprocessed = [line.replace('\n', ' ') for line in f.readlines()]
        keywords_not_preprocessed = [line for line in keywords_not_preprocessed if line not in (' ', '')]

    fuzzy: FuzzySearcher = FuzzySearcher(ratio=fuzz.token_sort_ratio,  # ratio=fuzz.ratio
                                         partial_ratio=fuzz.partial_ratio,
                                         conf_threshold_percent=80,
                                         preprocess=dummy_preprocess,
                                         keywords=keywords_not_preprocessed,
                                         log_path=project_dir / 'log.txt')

    # use single processor
    # result_xlsx: pd.DataFrame = fuzzy.try_concat_result((fuzzy.search_in_xlsx(xlsx_path)
    #                                                      for xlsx_path in xlsx_dir.glob('*.xlsx')))
    #
    # result_pdf: pd.DataFrame = fuzzy.try_concat_result((fuzzy.search_in_pdf(pdf_path)
    #                                                     for pdf_path in searchable_pdf_dir.glob('*.pdf')))

    with Pool(processes=1) as pool:
        result_xlsx: pd.DataFrame = fuzzy.try_concat_result(pool.map(fuzzy.search_in_xlsx,
                                                                     xlsx_dir.glob('*.xlsx')))
        result_pdf: pd.DataFrame = fuzzy.try_concat_result(pool.map(fuzzy.search_in_pdf,
                                                                    searchable_pdf_dir.glob('*.pdf')))

    print(f'\n{"* " * 35}\n')
    print(result_xlsx)
    print(f'\n{"* " * 35}\n')
    print(result_pdf)
