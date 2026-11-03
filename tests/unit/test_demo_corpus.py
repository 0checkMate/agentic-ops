"""RAG corpus seed: structure, supersession, and grounding of the ground truth."""

from collections import Counter

from agentic_ops.erp.mock.demo_seed import build_demo_dataset
from agentic_ops.rag.demo_corpus import (
    ABSTENTION_ANSWER,
    DEMO_DOCUMENTS,
    DEMO_QUESTIONS,
    DocStatus,
    DocType,
    QuestionKind,
)


def _texts() -> dict[str, str]:
    return {passage.passage_id: passage.text for doc in DEMO_DOCUMENTS for passage in doc.passages}


def _document_of(passage_id: str) -> str:
    return passage_id.split('#')[0]


def test_ids_are_unique() -> None:
    doc_ids = [doc.doc_id for doc in DEMO_DOCUMENTS]
    passage_ids = [p.passage_id for doc in DEMO_DOCUMENTS for p in doc.passages]
    question_ids = [question.question_id for question in DEMO_QUESTIONS]
    assert len(doc_ids) == len(set(doc_ids))
    assert len(passage_ids) == len(set(passage_ids))
    assert len(question_ids) == len(set(question_ids))


def test_every_document_type_is_present() -> None:
    assert {doc.doc_type for doc in DEMO_DOCUMENTS} == set(DocType)


def test_passages_are_chunk_sized_and_namespaced() -> None:
    for doc in DEMO_DOCUMENTS:
        assert len(doc.passages) >= 3
        for passage in doc.passages:
            assert passage.passage_id.startswith(f'{doc.doc_id}#')
            assert 120 <= len(passage.text) <= 1200


def test_superseded_documents_have_one_current_successor() -> None:
    by_id = {doc.doc_id: doc for doc in DEMO_DOCUMENTS}
    assert all(doc.supersedes in by_id for doc in DEMO_DOCUMENTS if doc.supersedes is not None)
    superseded = [doc for doc in DEMO_DOCUMENTS if doc.status is DocStatus.SUPERSEDED]
    assert len(superseded) >= 3
    for old in superseded:
        successors = [doc for doc in DEMO_DOCUMENTS if doc.supersedes == old.doc_id]
        assert len(successors) == 1
        new = successors[0]
        assert new.status is DocStatus.CURRENT
        assert new.doc_code == old.doc_code
        assert new.revision > old.revision
        assert new.effective_date > old.effective_date
    current_per_code = Counter(
        doc.doc_code for doc in DEMO_DOCUMENTS if doc.status is DocStatus.CURRENT
    )
    assert all(count == 1 for count in current_per_code.values())


def test_answerable_questions_are_grounded() -> None:
    texts = _texts()
    for question in DEMO_QUESTIONS:
        if question.kind is QuestionKind.UNANSWERABLE:
            continue
        assert question.source_passage_ids
        assert question.required_facts
        assert all(pid in texts for pid in question.source_passage_ids)
        cited = ' '.join(texts[pid] for pid in question.source_passage_ids).lower()
        for fact in question.required_facts:
            assert fact.lower() in cited, (question.question_id, fact)


def test_unanswerable_questions_expect_abstention() -> None:
    unanswerable = [q for q in DEMO_QUESTIONS if q.kind is QuestionKind.UNANSWERABLE]
    assert unanswerable
    for question in unanswerable:
        assert question.source_passage_ids == ()
        assert question.required_facts == ()
        assert question.stale_passage_ids == ()
        assert question.expected_answer == ABSTENTION_ANSWER


def test_multi_hop_questions_span_documents() -> None:
    multi_hop = [q for q in DEMO_QUESTIONS if q.kind is QuestionKind.MULTI_HOP]
    assert multi_hop
    for question in multi_hop:
        owners = {_document_of(pid) for pid in question.source_passage_ids}
        assert len(owners) >= 2


def test_superseded_traps_discriminate_current_from_stale() -> None:
    texts = _texts()
    owner = {p.passage_id: doc for doc in DEMO_DOCUMENTS for p in doc.passages}
    for question in DEMO_QUESTIONS:
        if question.kind is not QuestionKind.SUPERSEDED:
            assert question.stale_passage_ids == ()
            continue
        assert question.stale_passage_ids
        assert all(owner[pid].status is DocStatus.SUPERSEDED for pid in question.stale_passage_ids)
        assert all(owner[pid].status is DocStatus.CURRENT for pid in question.source_passage_ids)
        stale_text = ' '.join(texts[pid] for pid in question.stale_passage_ids).lower()
        for fact in question.required_facts:
            assert fact.lower() not in stale_text, (question.question_id, fact)


def test_question_mix_meets_minimums() -> None:
    counts = Counter(question.kind for question in DEMO_QUESTIONS)
    assert counts[QuestionKind.SINGLE] >= 10
    assert counts[QuestionKind.MULTI_HOP] >= 4
    assert counts[QuestionKind.SUPERSEDED] >= 3
    assert counts[QuestionKind.UNANSWERABLE] >= 4
    assert len(DEMO_QUESTIONS) >= 21


def test_spec_documents_point_at_real_erp_products() -> None:
    product_codes = {product.default_code for product in build_demo_dataset().products}
    for doc in DEMO_DOCUMENTS:
        if doc.doc_type is DocType.SPEC:
            assert doc.product_code in product_codes
        else:
            assert doc.product_code is None
