class RagError(Exception):
    pass


class DocumentParseError(RagError):
    pass


class EmptyDocumentError(RagError):
    pass


class MissingAPIKeyError(RagError):
    pass
