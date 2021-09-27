"""
Class for OCR scanned pdf
"""
# pylint: disable=line-too-long

import tempfile
from pathlib import Path
from multiprocessing import Pool
from typing import Dict, List, Any
import datetime

import cv2
import numpy as np
from PIL import Image
import pytesseract

import fitz
import pdf2image


class Recognizer:
    """
    Basic class to OCR scanned pdf
    """

    def __init__(self, dpi: int, log_path: Path, searchable_pdf_dir: Path, preprocess_config: Dict, lang: str) -> None:
        """
        :param dpi: Dots per inch, dpi >= 300 recommended
        :param log_path: path to log
        :param searchable_pdf_dir: directory to save pdf after OCR
        :param preprocess_config: config with keys 'resize', 'adaptiveThreshold', 'bilateralFilter'
        :param lang: lang for tesseract-osr
        """
        # pylint: disable=too-many-arguments

        self.dpi = int(dpi)
        self.preprocess_config = preprocess_config
        self.lang = lang
        self.log_path = log_path
        self.searchable_pdf_dir = searchable_pdf_dir
        self.log(f'Recognizer initialization: {str(datetime.datetime.now())}')

    def log(self, *args, **kwargs) -> None:
        """
        Write log to self.log file
        """
        print(*args, **kwargs)
        with open(self.log_path, 'a', encoding='utf-8') as file:
            print(*args, **kwargs, file=file, flush=True)

    # @staticmethod
    # def get_text_percentage_page(pdf_page: fitz.Page) -> float:
    #     """
    #     Calculate the percentage of document that is covered by (searchable) text.
    #     https://stackoverflow.com/questions/55704218/how-to-check-if-pdf-is-scanned-image-or-contains-text
    #
    #     If the returned percentage of text is very low, the document is
    #     most likely a scanned PDF
    #     """
    #     total_page_area = abs(pdf_page.rect)
    #     total_text_area = 0.0
    #
    #     for b in pdf_page.get_text(opt="blocks"):
    #         r = fitz.Rect(b[:4])  # rectangle where block text appears
    #         total_text_area += abs(r)
    #     return total_text_area / total_page_area

    @staticmethod
    def image_preprocess(image: Image, config: Dict) -> Any:
        """
        Preprocess image to best recognition
        :param image: PIL.Image form pdf2image.convert_from_path
        :param config: config with keys 'resize', 'adaptiveThreshold', 'bilateralFilter'
        :return: img as np.array
        """
        img = np.array(image.convert('RGB'))[:, :, ::-1].copy()
        if config.get('resize', False):
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        if config.get('adaptiveThreshold', False):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        if config.get('bilateralFilter', False):
            img = cv2.bilateralFilter(img, 9, 75, 75)
        return img

    def scanned2searchable(self, pdf_path: Path) -> None:
        """
        Use pdf2image and tesseract to convert scanned pdf to searchable
        :param pdf_path: path to one pdf file
        :return: None, pdf save to searchable pdf dir
        """
        with tempfile.TemporaryDirectory(prefix=pdf_path.name) as tmp:
            self.log(f'Start pdf recognition: {pdf_path.name}')
            pdf: fitz.Document = fitz.Document()
            images: List[Image] = pdf2image.convert_from_path(pdf_path, output_folder=tmp, dpi=self.dpi)
            for page_num, img in enumerate(images):
                if self.preprocess_config:
                    img: np.array = self.image_preprocess(image=img, config=self.preprocess_config)
                try:
                    page = pytesseract.image_to_pdf_or_hocr(img, lang=self.lang, config='--psm 6')
                    with fitz.open('pdf', page) as page:
                        pdf.insert_pdf(page)
                except Exception as ex:
                    self.log(f'{ex} on page {page_num} in pdf file {pdf_path.name}')
            self.log(f'Done pdf recognition: {pdf_path.name}')
            pdf.save(self.searchable_pdf_dir / pdf_path.name)


if __name__ == '__main__':
    # export TESSDATA_PREFIX='/usr/share/tesseract-ocr/4.00/tessdata'
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

    project_dir = Path.cwd().parent
    scanned_pdf_dir = project_dir / 'inp' / 'scanned pdf'

    recognizer = Recognizer(dpi=600, log_path=project_dir / 'log.txt', lang='ru',
                            searchable_pdf_dir=project_dir / 'inp' / 'searchable pdf',
                            preprocess_config={'resize': False, 'adaptiveThreshold': False, 'bilateralFilter': False})

    with Pool(processes=4) as pool:
        pool.map(recognizer.scanned2searchable, scanned_pdf_dir.glob('*.pdf'))

    recognizer.log(f'Recognizer finish: {str(datetime.datetime.now())}')
