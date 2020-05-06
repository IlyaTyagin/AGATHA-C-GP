from agatha.construct import semrep_util
from pathlib import Path
import pytest
import lxml
import dask.bag as dbag
import agatha.construct.dask_process_global as dpg


# If SemRep isn't present, don't bother with these tests
SEMREP_INSTALL_DIR = Path("externals/semrep/2020/public_semrep")
METAMAP_INSTALL_DIR = Path("externals/semrep/2020/public_mm")
UNICODE_TO_ASCII_JAR_PATH = Path("externals/semrep/replace_utf8.jar")
TEST_DATA_PATH = Path("test_data/semrep_input.txt")
TEST_COVID_DATA_PATH = Path("test_data/semrep_covid_input.txt")
TEST_SEMREP_SEGFAULT_PATH= Path("test_data/semrep_segfault_input.txt")
TEST_COVID_XML_PATH = Path("test_data/semrep_covid.xml")

RUN_SEMREP_TESTS = (
    SEMREP_INSTALL_DIR.is_dir()
    and METAMAP_INSTALL_DIR.is_dir()
    and UNICODE_TO_ASCII_JAR_PATH.is_file()
)


def test_unicode_to_ascii():
  if UNICODE_TO_ASCII_JAR_PATH.is_file():
    text_in = ["α", "β"]
    expected = ["alpha", "beta"]
    u2i = semrep_util.UnicodeToAsciiRunner(UNICODE_TO_ASCII_JAR_PATH)
    assert u2i(text_in) == expected

## TESTS THAT REQUIRE ALL OF SEMREP BELOW HERE

if RUN_SEMREP_TESTS:
  print("STARTING METAMAP SERVER")
  # The metamap server takes about 30 seconds to initialize, so we'll just do
  # that once.
  metamap_server = semrep_util.MetaMapServer(METAMAP_INSTALL_DIR)
  metamap_server.start()

def test_get_all_paths():
  "Tests that getting semrep paths gets all needed paths"
  if RUN_SEMREP_TESTS:
    paths = semrep_util.get_paths(
        semrep_install_dir=SEMREP_INSTALL_DIR,
        metamap_install_dir=METAMAP_INSTALL_DIR
    )
    assert "metamap_install_dir" in paths
    assert "metamap_pos_server_path" in paths
    assert "metamap_wsd_server_path" in paths
    assert "semrep_install_dir" in paths
    assert "semrep_lib_dir" in paths
    assert "semrep_preamble_path" in paths
    assert "semrep_bin_path" in paths

def test_semrep_paths():
  "Tests that if we just need the semrep paths, we can get those"
  if RUN_SEMREP_TESTS:
    paths = semrep_util.get_paths(
        semrep_install_dir=SEMREP_INSTALL_DIR,
    )
    assert "semrep_install_dir" in paths
    assert "semrep_lib_dir" in paths
    assert "semrep_preamble_path" in paths
    assert "semrep_bin_path" in paths

def test_get_metamap_paths():
  "Tests that getting semrep paths gets all needed paths"
  if RUN_SEMREP_TESTS:
    paths = semrep_util.get_paths(
        metamap_install_dir=METAMAP_INSTALL_DIR
    )
    assert "metamap_install_dir" in paths
    assert "metamap_pos_server_path" in paths
    assert "metamap_wsd_server_path" in paths

def test_get_semrep_paths_fails():
  "Tests that if you give semrep paths bad install locations, it fails"
  if RUN_SEMREP_TESTS:
    with pytest.raises(AssertionError):
      semrep_util.get_paths(SEMREP_INSTALL_DIR, Path("."))
    with pytest.raises(AssertionError):
      semrep_util.get_paths(Path("."), METAMAP_INSTALL_DIR)
    with pytest.raises(AssertionError):
      semrep_util.get_paths(Path("."), Path("."))
    with pytest.raises(AssertionError):
      semrep_util.get_paths(semrep_install_dir=Path("."))
      semrep_util.get_paths(metamap_install_dir=Path("."))

def test_metamap_server():
  "Tests that we can actually run metamap"
  if RUN_SEMREP_TESTS:
    assert metamap_server.running()

def test_run_semrep():
  if RUN_SEMREP_TESTS:
    runner = semrep_util.SemRepRunner(
        semrep_install_dir=SEMREP_INSTALL_DIR,
        metamap_server=metamap_server,
        lexicon_year=2020,
        mm_data_year="2020AA",
    )
    output_file = Path("/tmp/test_run_semrep.xml")
    if output_file.is_file():
      output_file.unlink()
    assert not output_file.exists()
    runner.run(TEST_DATA_PATH, output_file)
    assert output_file.is_file()

def test_run_semrep_covid():
  if RUN_SEMREP_TESTS:
    runner = semrep_util.SemRepRunner(
        semrep_install_dir=SEMREP_INSTALL_DIR,
        metamap_server=metamap_server,
        lexicon_year=2020,
        mm_data_year="2020AA",
    )
    output_file = Path("/tmp/test_run_semrep_covid.xml")
    if output_file.is_file():
      output_file.unlink()
    assert not output_file.exists()
    runner.run(TEST_COVID_DATA_PATH, output_file)
    assert output_file.is_file()

def test_sentence_to_semrep_input():
  # We can run this test if SemRep is not installed
  sentences = [
      dict(id=1, sent_text="Sentence 1"),
      dict(id=2, sent_text="Sentence 2"),
  ]

  actual = semrep_util.sentences_to_semrep_input(
      sentences,
      UNICODE_TO_ASCII_JAR_PATH,
  )
  expected = ["1|Sentence 1", "2|Sentence 2"]
  assert actual == expected

def test_sentence_to_semrep_input_filter_newline():
  # We can run this test if SemRep is not installed
  sentences = [
      dict(id=1, sent_text="Sentence\n1"),
      dict(id=2, sent_text="Sentence\n2"),
  ]

  actual = semrep_util.sentences_to_semrep_input(
      sentences,
      UNICODE_TO_ASCII_JAR_PATH
  )
  expected = ["1|Sentence 1", "2|Sentence 2"]
  assert actual == expected

def test_sentence_to_semrep_input_filter_unicode():
  # We can run this test if SemRep is not installed
  sentences = [
      dict(id=1, sent_text="Sentence α"),
      dict(id=2, sent_text="Sentence β"),
  ]
  actual = semrep_util.sentences_to_semrep_input(
      sentences,
      UNICODE_TO_ASCII_JAR_PATH
  )
  expected = ["1|Sentence alpha", "2|Sentence beta"]
  assert actual == expected

def test_semrep_xml_to_records():
  "Ensures that parsing xml files happens without error"
  predicates = semrep_util.semrep_xml_to_records(TEST_COVID_XML_PATH)
  assert len(predicates) > 0

def test_semrep_id_to_agatha_sentence_id():
  expected = "s:12345:1:12"
  assert semrep_util._semrep_id_to_agatha_sentence_id(
      "Ds:12345:1:12"
  ) == expected
  assert semrep_util._semrep_id_to_agatha_sentence_id(
      "Ds:12345:1:12.E2"
  ) == expected
  assert semrep_util._semrep_id_to_agatha_sentence_id(
      "Ds:12345:1:12.P11"
  ) == expected
  assert semrep_util._semrep_id_to_agatha_sentence_id(
      "Ds:12345:1:12.tx.2"
  ) == expected

def test_semrep_id_to_agatha_sentence_id_weird_id():
  expected = "s:abcd123efg567:1:12"
  assert semrep_util._semrep_id_to_agatha_sentence_id(
      "Ds:abcd123efg567:1:12"
  ) == expected
  assert semrep_util._semrep_id_to_agatha_sentence_id(
      "Ds:abcd123efg567:1:12.E2"
  ) == expected
  assert semrep_util._semrep_id_to_agatha_sentence_id(
      "Ds:abcd123efg567:1:12.P11"
  ) == expected
  assert semrep_util._semrep_id_to_agatha_sentence_id(
      "Ds:abcd123efg567:1:12.tx.2"
  ) == expected


def test_parse_semrep_xml_entity():
  raw_xml_entity = """
    <Entity
      id="Ds:32353859:1:6.E1"
      cui="C1517331"
      name="Further"
      semtypes="spco"
      text="Further"
      score="888"
      negated="false"
      begin="0"
      end="7"
    />
  """
  xml_entity = lxml.etree.fromstring(raw_xml_entity)
  expected = dict(
      id="s:32353859:1:6",
      cui="C1517331",
      name="Further",
      score=888,
      negated=False,
      begin=0,
      end=7,
  )
  assert semrep_util._parse_semrep_xml_entity(xml_entity) == expected

def test_parse_semrep_xml_predication():
  raw_xml_predication = """
  <Predication id="Ds:123:1:2.P1" negated="true" inferred="false">
   <Subject maxDist="5" dist="1" entityID="Ds:123:1:2.E1" relSemType="topp" />
   <Predicate type="USES" indicatorType="PREP" begin="1" end="2" />
   <Object maxDist="3" dist="1" entityID="Ds:123:1:2.E2" relSemType="aapp" />
  </Predication>
  """
  xml_predication= lxml.etree.fromstring(raw_xml_predication)
  semrepid2entity = {
      "Ds:123:1:2.E1": {
        "cui": "C1",
        "name": "First",
        "score": 888,
        "negated": False,
        "begin": 0,
        "end": 7,
      },
      "Ds:123:1:2.E2": {
        "cui": "C2",
        "name": "Second",
        "score": 888,
        "negated": False,
        "begin": 9,
        "end": 13,
      },
  }
  expected = {
      "negated": True,
      "inferred": False,
      "subject": {
        "cui": "C1",
        "name": "First",
        "score": 888,
        "negated": False,
        "begin": 0,
        "end": 7,
        "maxDist": 5,
        "dist": 1,
        "relSemType": "topp",
      },
      "predicate": {
        "type": "USES",
        "indicatorType": "PREP",
        "begin": 1,
        "end": 2,
      },
      "object": {
        "cui": "C2",
        "name": "Second",
        "score": 888,
        "negated": False,
        "begin": 9,
        "end": 13,
        "maxDist": 3,
        "dist": 1,
        "relSemType": "aapp",
      },
  }
  assert semrep_util._parse_semrep_xml_predication(
      xml_predication, semrepid2entity
  ) == expected


def test_parse_semrep_end_to_end():
  # Run SemRep
  if RUN_SEMREP_TESTS:
    records = [
        {
          "id": "s:1234:1:2",
          "sent_text": "Tobacco causes cancer in mice."
        },
        {
          "id": "s:2345:1:2",
          "sent_text": "Tobacco causes cancer in humans."
        },
    ]

    tmp_semrep_input = Path("/tmp/test_parse_semrep_end_to_end_input")
    tmp_semrep_output = Path("/tmp/test_parse_semrep_end_to_end_output")
    if tmp_semrep_input.is_file():
      tmp_semrep_input.unlink()
    if tmp_semrep_output.is_file():
      tmp_semrep_output.unlink()

    with open(tmp_semrep_input, 'w') as semrep_input_file:
      for line in semrep_util.sentences_to_semrep_input(
          records,
          UNICODE_TO_ASCII_JAR_PATH
      ):
        semrep_input_file.write(f"{line}\n")

    runner = semrep_util.SemRepRunner(
        semrep_install_dir=SEMREP_INSTALL_DIR,
        metamap_server=metamap_server,
        lexicon_year=2020,
        mm_data_year="2020AA",
    )
    runner.run(tmp_semrep_input, tmp_semrep_output)
    assert tmp_semrep_output.is_file()

    # should return one per document
    records = semrep_util.semrep_xml_to_records(tmp_semrep_output)
    assert len(records) == 2

def test_extract_entitites_and_predicates_with_dask():
  if RUN_SEMREP_TESTS:
    records = dbag.from_sequence([
        {
          "id": "s:1234:1:2",
          "sent_text": "Tobacco causes cancer in mice."
        },
        {
          "id": "s:2345:1:2",
          "sent_text": "Tobacco causes cancer in humans."
        },
    ], npartitions=1)
    work_dir = Path("/tmp/test_extract_entitites_and_predicates_with_dask")
    work_dir.mkdir(exist_ok=True, parents=True)
    # Configure Metamap Server through DPG
    preloader = dpg.WorkerPreloader()
    preloader.register(*semrep_util.get_metamap_server_initializer(
      metamap_install_dir=METAMAP_INSTALL_DIR,
    ))
    dpg.add_global_preloader(preloader=preloader)

    actual = semrep_util.extract_entities_and_predicates_from_sentences(
        sentence_records=records,
        unicode_to_ascii_jar_path=UNICODE_TO_ASCII_JAR_PATH,
        semrep_install_dir=SEMREP_INSTALL_DIR,
        work_dir=work_dir,
        lexicon_year=2020,
        mm_data_year="2020AA",
    ).compute()
    assert len(actual) == 2

