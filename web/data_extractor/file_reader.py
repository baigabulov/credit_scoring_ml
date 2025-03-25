import mimetypes

import tiktoken
from docx import Document

from django.conf import settings


def read_file(file):

    # Check file type, because supports only text or docx formats
    file_type, encoding = mimetypes.guess_type(file.name)

    if file_type == 'text/plain':
        content = file.file.read()
        return content.decode('utf-8')

    elif file_type in [
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ]:
        document = Document(file.file)
        text = []
        for paragraph in document.paragraphs:
            text.append(paragraph.text)

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
