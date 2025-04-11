import mimetypes

import tiktoken
from docx import Document
import fitz

from django.conf import settings


def read_file(file):

    # Check file type, because supports only text or docx or pdf formats
    file_type, encoding = mimetypes.guess_type(file.name)

    if file_type == 'text/plain':
        content = file.file.read()
        print('text file content', content)
        return content.decode('utf-8')

    elif file_type in [
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ]:
        document = Document(file.file)
        text = []
        for paragraph in document.paragraphs:
            text.append(paragraph.text)

        print('docx file content', '\n'.join(text))
        return '\n'.join(text)
    
    elif file_type == 'application/pdf':
        text = []
        doc = fitz.open(stream=file.file.read(), filetype="pdf")
        for page in doc:
            text.append(page.get_text())
        doc.close()
        print('pdf file content', '\n'.join(text))
        return '\n'.join(text)

    else:
        raise Exception('unsupported type of file %s' % file_type)


def tokenize(text):
    model = settings.CHATGPT_MODEL
    tokenizer = tiktoken.encoding_for_model(model)
    tokenized_text = tokenizer.encode(text)
    if len(tokenized_text) > settings.CHATGPT_TOKEN_MAX_LENGTH:
        raise Exception('token max length reached, try with fewer words')
    return tokenized_text
