# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: agatha/ml/abstract_generator/sentencepiece.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='agatha/ml/abstract_generator/sentencepiece.proto',
  package='sentencepiece',
  syntax='proto2',
  serialized_options=_b('H\003'),
  serialized_pb=_b('\n0agatha/ml/abstract_generator/sentencepiece.proto\x12\rsentencepiece\"\xdf\x01\n\x11SentencePieceText\x12\x0c\n\x04text\x18\x01 \x01(\t\x12>\n\x06pieces\x18\x02 \x03(\x0b\x32..sentencepiece.SentencePieceText.SentencePiece\x12\r\n\x05score\x18\x03 \x01(\x02\x1a\x62\n\rSentencePiece\x12\r\n\x05piece\x18\x01 \x01(\t\x12\n\n\x02id\x18\x02 \x01(\r\x12\x0f\n\x07surface\x18\x03 \x01(\t\x12\r\n\x05\x62\x65gin\x18\x04 \x01(\r\x12\x0b\n\x03\x65nd\x18\x05 \x01(\r*\t\x08\xc8\x01\x10\x80\x80\x80\x80\x02*\t\x08\xc8\x01\x10\x80\x80\x80\x80\x02\"J\n\x16NBestSentencePieceText\x12\x30\n\x06nbests\x18\x01 \x03(\x0b\x32 .sentencepiece.SentencePieceTextB\x02H\x03')
)




_SENTENCEPIECETEXT_SENTENCEPIECE = _descriptor.Descriptor(
  name='SentencePiece',
  full_name='sentencepiece.SentencePieceText.SentencePiece',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='piece', full_name='sentencepiece.SentencePieceText.SentencePiece.piece', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='id', full_name='sentencepiece.SentencePieceText.SentencePiece.id', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='surface', full_name='sentencepiece.SentencePieceText.SentencePiece.surface', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='begin', full_name='sentencepiece.SentencePieceText.SentencePiece.begin', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='end', full_name='sentencepiece.SentencePieceText.SentencePiece.end', index=4,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=True,
  syntax='proto2',
  extension_ranges=[(200, 536870912), ],
  oneofs=[
  ],
  serialized_start=182,
  serialized_end=280,
)

_SENTENCEPIECETEXT = _descriptor.Descriptor(
  name='SentencePieceText',
  full_name='sentencepiece.SentencePieceText',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='text', full_name='sentencepiece.SentencePieceText.text', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pieces', full_name='sentencepiece.SentencePieceText.pieces', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='score', full_name='sentencepiece.SentencePieceText.score', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_SENTENCEPIECETEXT_SENTENCEPIECE, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=True,
  syntax='proto2',
  extension_ranges=[(200, 536870912), ],
  oneofs=[
  ],
  serialized_start=68,
  serialized_end=291,
)


_NBESTSENTENCEPIECETEXT = _descriptor.Descriptor(
  name='NBestSentencePieceText',
  full_name='sentencepiece.NBestSentencePieceText',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='nbests', full_name='sentencepiece.NBestSentencePieceText.nbests', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=293,
  serialized_end=367,
)

_SENTENCEPIECETEXT_SENTENCEPIECE.containing_type = _SENTENCEPIECETEXT
_SENTENCEPIECETEXT.fields_by_name['pieces'].message_type = _SENTENCEPIECETEXT_SENTENCEPIECE
_NBESTSENTENCEPIECETEXT.fields_by_name['nbests'].message_type = _SENTENCEPIECETEXT
DESCRIPTOR.message_types_by_name['SentencePieceText'] = _SENTENCEPIECETEXT
DESCRIPTOR.message_types_by_name['NBestSentencePieceText'] = _NBESTSENTENCEPIECETEXT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SentencePieceText = _reflection.GeneratedProtocolMessageType('SentencePieceText', (_message.Message,), dict(

  SentencePiece = _reflection.GeneratedProtocolMessageType('SentencePiece', (_message.Message,), dict(
    DESCRIPTOR = _SENTENCEPIECETEXT_SENTENCEPIECE,
    __module__ = 'agatha.ml.abstract_generator.sentencepiece_pb2'
    # @@protoc_insertion_point(class_scope:sentencepiece.SentencePieceText.SentencePiece)
    ))
  ,
  DESCRIPTOR = _SENTENCEPIECETEXT,
  __module__ = 'agatha.ml.abstract_generator.sentencepiece_pb2'
  # @@protoc_insertion_point(class_scope:sentencepiece.SentencePieceText)
  ))
_sym_db.RegisterMessage(SentencePieceText)
_sym_db.RegisterMessage(SentencePieceText.SentencePiece)

NBestSentencePieceText = _reflection.GeneratedProtocolMessageType('NBestSentencePieceText', (_message.Message,), dict(
  DESCRIPTOR = _NBESTSENTENCEPIECETEXT,
  __module__ = 'agatha.ml.abstract_generator.sentencepiece_pb2'
  # @@protoc_insertion_point(class_scope:sentencepiece.NBestSentencePieceText)
  ))
_sym_db.RegisterMessage(NBestSentencePieceText)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)