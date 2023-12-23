import os

from io import StringIO
from typing import Tuple

import numpy as np

from pdfminer.converter import TextConverter
from pdfminer.high_level import extract_pages
from pdfminer.image import ImageWriter
from pdfminer.layout import LAParams, LTContainer, LTFigure, LTImage, LTTextContainer
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser


class CustomImageWriter(ImageWriter):
    def _create_unique_image_name(self, image: LTImage, ext: str) -> Tuple[str, str]:
        name = "_".join(str(int(b)) for b in image.bbox) + ext
        path = os.path.join(self.outdir, name)
        img_index = 0
        while os.path.exists(path):
            name = "%s.%d%s" % (image.name, img_index, ext)
            path = os.path.join(self.outdir, name)
            img_index += 1
        return name, path


def pdf_extract_text(path):
    # extract text and images
    output_string = StringIO()
    with open(path, "rb") as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)

    return output_string.getvalue()


def pdf_extract_text_figures(path, image_output_path):
    os.makedirs(image_output_path, exist_ok=True)
    iw = CustomImageWriter(image_output_path)

    # extract text and images
    output_string = StringIO()
    with open(path, "rb") as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(
            rsrcmgr, output_string, laparams=LAParams(), imagewriter=iw
        )
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)

    # match images with text
    figures = []
    current_figure = [None, None, []]

    def iterate_container(element):
        global current_figure
        if isinstance(element, LTTextContainer):
            if "Figure" in element.get_text():
                if current_figure[0] is not None:
                    figures.append(
                        (current_figure[0], current_figure[1], current_figure[2])
                    )
                    current_figure = [element.bbox, element.get_text()[:9], []]

                elif current_figure[0] is None and current_figure[2] != []:
                    current_figure[0] = element.bbox
                    current_figure[1] = element.get_text()[:9]
                    figures.append(
                        (current_figure[0], current_figure[1], current_figure[2])
                    )
                    current_figure = [None, None, []]

                else:
                    current_figure[0] = element.bbox
                    current_figure[1] = element.get_text()[:9]

            return element.get_text()
        elif isinstance(element, LTImage) or isinstance(element, LTFigure):
            current_figure[2].append(element.bbox)

        elif isinstance(element, LTContainer):
            for child in element:
                return iterate_container(child, current_figure)

    for page_layout in extract_pages(path):
        for element in page_layout:
            iterate_container(element)

    if current_figure[2] != []:
        figures.append(current_figure)

    image_files = os.listdir(image_output_path)
    for figure in figures:
        for i, bbox in enumerate(figure[2]):
            bbox_round = np.array([int(b) for b in bbox])
            for image_file in image_files:
                image_bbox = np.array(
                    [int(b) for b in image_file.split(".")[0].split("_")]
                )
                if np.sum(np.abs(bbox_round - image_bbox)) < 5:
                    os.rename(
                        os.path.join(image_output_path, image_file),
                        os.path.join(
                            image_output_path,
                            figure[1] + "_" + str(i) + "." + image_file.split(".")[-1],
                        ),
                    )
                    break

    return output_string.getvalue(), os.listdir(image_output_path)
