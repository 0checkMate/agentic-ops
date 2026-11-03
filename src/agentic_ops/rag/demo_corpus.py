"""Synthetic quality-document corpus with a RAG answer key.

All documents and values are invented for evaluation. They are not food-safety guidance.
Every answerable question lists the passages that hold the answer and the exact facts that
must appear in it, so retrieval, faithfulness and citation metrics can be scored by machine.
"""

from dataclasses import dataclass
from datetime import date
from enum import StrEnum

ABSTENTION_ANSWER = 'The documents do not contain this information.'


class DocType(StrEnum):
    SOP = 'sop'
    SPEC = 'spec'
    HACCP = 'haccp'


class DocStatus(StrEnum):
    CURRENT = 'current'
    SUPERSEDED = 'superseded'


class QuestionKind(StrEnum):
    SINGLE = 'single'
    MULTI_HOP = 'multi_hop'
    SUPERSEDED = 'superseded'
    UNANSWERABLE = 'unanswerable'


@dataclass(frozen=True)
class Passage:
    passage_id: str
    text: str


@dataclass(frozen=True)
class CorpusDocument:
    doc_id: str
    doc_code: str
    title: str
    doc_type: DocType
    revision: int
    status: DocStatus
    effective_date: date
    passages: tuple[Passage, ...]
    supersedes: str | None = None
    product_code: str | None = None


@dataclass(frozen=True)
class CorpusQuestion:
    question_id: str
    kind: QuestionKind
    question: str
    expected_answer: str
    source_passage_ids: tuple[str, ...] = ()
    required_facts: tuple[str, ...] = ()
    stale_passage_ids: tuple[str, ...] = ()


def _passages(doc_id: str, *texts: str) -> tuple[Passage, ...]:
    return tuple(
        Passage(passage_id=f'{doc_id}#p{number}', text=text)
        for number, text in enumerate(texts, start=1)
    )


_RECEIVING_R1 = 'SOP-QA-014-R1'
_RECEIVING_R2 = 'SOP-QA-014-R2'
_CIP_R1 = 'SOP-PR-021-R1'
_CIP_R2 = 'SOP-PR-021-R2'
_HACCP_R1 = 'HACCP-001-R1'
_HACCP_R2 = 'HACCP-001-R2'
_MANGO_R1 = 'SPEC-ING-001-R1'
_MANGO_R2 = 'SPEC-ING-001-R2'

DEMO_DOCUMENTS: tuple[CorpusDocument, ...] = (
    CorpusDocument(
        doc_id=_RECEIVING_R1,
        doc_code='SOP-QA-014',
        title='Receiving of chilled produce',
        doc_type=DocType.SOP,
        revision=1,
        status=DocStatus.SUPERSEDED,
        effective_date=date(2025, 3, 1),
        passages=_passages(
            _RECEIVING_R1,
            (
                'Scope: this procedure applies to all chilled fruit and vegetables delivered to '
                'the intake dock. Every delivery is checked by a QA technician before stores '
                'signs for it.'
            ),
            (
                'Temperature: the core temperature of each delivery is measured with a calibrated '
                'probe. A delivery is accepted when the core temperature is 8 °C or lower. '
                'Deliveries above 8 °C are held at the dock and QA is notified.'
            ),
            (
                'Rejection: a delivery that fails the temperature check or shows visible mould, '
                'soil contamination or pest damage is rejected and returned to the supplier. '
                'The rejection is recorded on form QA-F-014 on the same day.'
            ),
        ),
    ),
    CorpusDocument(
        doc_id=_RECEIVING_R2,
        doc_code='SOP-QA-014',
        title='Receiving of chilled produce',
        doc_type=DocType.SOP,
        revision=2,
        status=DocStatus.CURRENT,
        effective_date=date(2026, 2, 1),
        supersedes=_RECEIVING_R1,
        passages=_passages(
            _RECEIVING_R2,
            (
                'Scope: this procedure applies to all chilled fruit and vegetables delivered to '
                'the intake dock. Every delivery is checked by a QA technician before stores '
                'signs for it.'
            ),
            (
                'Temperature: the core temperature is measured with a calibrated probe, with one '
                'reading per pallet. A delivery is accepted when the core temperature is 6 °C or '
                'lower. Deliveries above 6 °C are held at the dock and QA is notified.'
            ),
            (
                'Release and rejection: deliveries between 6 °C and 10 °C may be released by the '
                'QA lead after a documented risk assessment. Deliveries above 10 °C are always '
                'rejected. Rejections are recorded on form QA-F-014 on the same day.'
            ),
        ),
    ),
    CorpusDocument(
        doc_id=_CIP_R1,
        doc_code='SOP-PR-021',
        title='Cleaning in place of the press and filling line',
        doc_type=DocType.SOP,
        revision=1,
        status=DocStatus.SUPERSEDED,
        effective_date=date(2025, 4, 15),
        passages=_passages(
            _CIP_R1,
            (
                'Scope: this procedure covers cleaning in place (CIP) of the cold press, the '
                'transfer lines and the filling line at the end of every production day, and '
                'after every change of product family.'
            ),
            (
                'Caustic wash: circulate a 1.5 % caustic soda solution at 70 °C for 20 minutes. '
                'The concentration is verified by conductivity at the start and the end of the '
                'wash.'
            ),
            (
                'Rinse and verify: rinse with potable water until the final rinse conductivity is '
                'below 200 µS/cm. An ATP swab is taken from the filler nozzle and the result is '
                'recorded on the CIP log.'
            ),
        ),
    ),
    CorpusDocument(
        doc_id=_CIP_R2,
        doc_code='SOP-PR-021',
        title='Cleaning in place of the press and filling line',
        doc_type=DocType.SOP,
        revision=2,
        status=DocStatus.CURRENT,
        effective_date=date(2026, 3, 1),
        supersedes=_CIP_R1,
        passages=_passages(
            _CIP_R2,
            (
                'Scope: this procedure covers cleaning in place (CIP) of the cold press, the '
                'transfer lines and the filling line at the end of every production day, and '
                'after every change of product family.'
            ),
            (
                'Caustic wash: circulate a 2.0 % caustic soda solution at 75 °C for 25 minutes. '
                'The concentration is verified by conductivity at the start and the end of the '
                'wash.'
            ),
            (
                'Rinse and verify: rinse with potable water until the final rinse conductivity is '
                'below 200 µS/cm. An ATP swab is taken from the filler nozzle. The result must be '
                '150 RLU or lower; above this limit the CIP cycle is repeated.'
            ),
        ),
    ),
    CorpusDocument(
        doc_id='SOP-QA-031-R1',
        doc_code='SOP-QA-031',
        title='Non-conformance and corrective action',
        doc_type=DocType.SOP,
        revision=1,
        status=DocStatus.CURRENT,
        effective_date=date(2026, 1, 15),
        passages=_passages(
            'SOP-QA-031-R1',
            (
                'A non-conformance record (NCR) is raised for every deviation from a '
                'specification, process limit or procedure. The NCR is opened within 24 hours of '
                'detection by the person who found the deviation.'
            ),
            (
                'Root cause analysis uses the 5-why method for minor events and a fishbone '
                'diagram for major or repeat events. A CAPA is opened when the root cause is '
                'confirmed, or when the same NCR category recurs three times in 90 days.'
            ),
            (
                'Corrective actions must be closed within 30 days of CAPA approval. An '
                'effectiveness check is performed 90 days after closure by the QA lead, and the '
                'CAPA is closed only if no recurrence is found.'
            ),
        ),
    ),
    CorpusDocument(
        doc_id='SOP-MT-008-R1',
        doc_code='SOP-MT-008',
        title='Preventive maintenance of the cold press',
        doc_type=DocType.SOP,
        revision=1,
        status=DocStatus.CURRENT,
        effective_date=date(2026, 1, 10),
        passages=_passages(
            'SOP-MT-008-R1',
            (
                'Scope: this procedure covers the hydraulic cold press and its feed conveyor. '
                'Maintenance is planned in the maintenance system and every task carries a '
                'work-order number.'
            ),
            (
                'The hydraulic oil filter is replaced every 500 operating hours and the press '
                'bags are replaced every 200 operating hours. Hydraulic oil is sampled for '
                'analysis every 1000 operating hours.'
            ),
            (
                'Pressure gauges are calibrated every 6 months against a reference gauge. A '
                'press that fails calibration is tagged out of service until the gauge is '
                'replaced and the QA lead releases it.'
            ),
        ),
    ),
    CorpusDocument(
        doc_id='SOP-QA-040-R1',
        doc_code='SOP-QA-040',
        title='Product recall and mock recall',
        doc_type=DocType.SOP,
        revision=1,
        status=DocStatus.CURRENT,
        effective_date=date(2026, 2, 15),
        passages=_passages(
            'SOP-QA-040-R1',
            (
                'A mock recall is carried out every 6 months on a randomly chosen raw-material '
                'lot. The exercise starts from the supplier lot number and must identify every '
                'finished lot and every customer that received it.'
            ),
            (
                'Targets: the full trace must be completed within 4 hours and must account for '
                '100 % of the produced quantity. The mass balance of produced, shipped and '
                'on-hand quantity must reconcile within 2 %.'
            ),
            (
                'Where an affected lot is partly on hand, the remaining stock is quarantined and '
                'its quantity is reported. Customers are notified by the QA lead using the '
                'template in form QA-F-040.'
            ),
        ),
    ),
    CorpusDocument(
        doc_id=_HACCP_R1,
        doc_code='HACCP-001',
        title='HACCP plan for cold-pressed juice',
        doc_type=DocType.HACCP,
        revision=1,
        status=DocStatus.SUPERSEDED,
        effective_date=date(2025, 6, 1),
        passages=_passages(
            _HACCP_R1,
            (
                'Product scope: cold-pressed fruit and vegetable juices packed in PET bottles '
                'and treated by high pressure processing (HPP) in the final pack. The product '
                'is distributed chilled.'
            ),
            (
                'CCP 1, high pressure processing: the critical limit is a minimum pressure of '
                '550 MPa held for 180 seconds. A batch that does not reach the limit is held and '
                'reprocessed or destroyed.'
            ),
            (
                'CCP 2, chilled storage: the critical limit is a maximum product temperature of '
                '4 °C. Temperature is logged every 15 minutes and deviations are escalated to '
                'the QA lead.'
            ),
        ),
    ),
    CorpusDocument(
        doc_id=_HACCP_R2,
        doc_code='HACCP-001',
        title='HACCP plan for cold-pressed juice',
        doc_type=DocType.HACCP,
        revision=2,
        status=DocStatus.CURRENT,
        effective_date=date(2026, 4, 1),
        supersedes=_HACCP_R1,
        passages=_passages(
            _HACCP_R2,
            (
                'Product scope: cold-pressed fruit and vegetable juices packed in PET bottles '
                'and treated by high pressure processing (HPP) in the final pack. The product '
                'is distributed chilled.'
            ),
            (
                'CCP 1, high pressure processing: the critical limit is a minimum pressure of '
                '600 MPa held for 180 seconds. A batch that does not reach the limit is held and '
                'reprocessed or destroyed.'
            ),
            (
                'CCP 2, chilled storage: the critical limit is a maximum product temperature of '
                '4 °C. Temperature is logged every 15 minutes and deviations are escalated to '
                'the QA lead.'
            ),
            (
                'CCP 3, foreign bodies: every filled bottle passes a metal detector. The detector '
                'is verified with test pieces of 2.0 mm ferrous, 2.5 mm non-ferrous and 3.0 mm '
                'stainless steel at start-up, every 2 hours and at the end of the run.'
            ),
            (
                'Corrective action for CCP 3: product packed since the last passed verification '
                'is placed on hold and re-inspected. The detector is not returned to service '
                'until it passes the full set of test pieces.'
            ),
        ),
    ),
    CorpusDocument(
        doc_id=_MANGO_R1,
        doc_code='SPEC-ING-001',
        title='Specification: mango puree',
        doc_type=DocType.SPEC,
        revision=1,
        status=DocStatus.SUPERSEDED,
        effective_date=date(2025, 5, 1),
        product_code='ING-001',
        passages=_passages(
            _MANGO_R1,
            (
                'Mango puree, pasteurised, supplied in aseptic drums. Appearance: uniform '
                'golden-yellow puree, free from fibre lumps, black specks and foreign matter.'
            ),
            (
                'Chemical: soluble solids of minimum 14 °Brix at 20 °C. The pH is between 3.4 and '
                '4.2. Both are measured on a composite sample from each delivered lot.'
            ),
            (
                'Microbiology: total plate count below 1000 CFU/g, yeast and mould below 100 '
                'CFU/g. Storage is ambient, below 25 °C, with a shelf life of 12 months.'
            ),
        ),
    ),
    CorpusDocument(
        doc_id=_MANGO_R2,
        doc_code='SPEC-ING-001',
        title='Specification: mango puree',
        doc_type=DocType.SPEC,
        revision=2,
        status=DocStatus.CURRENT,
        effective_date=date(2026, 3, 15),
        supersedes=_MANGO_R1,
        product_code='ING-001',
        passages=_passages(
            _MANGO_R2,
            (
                'Mango puree, pasteurised, supplied in aseptic drums. Appearance: uniform '
                'golden-yellow puree, free from fibre lumps, black specks and foreign matter.'
            ),
            (
                'Chemical: soluble solids of minimum 15 °Brix at 20 °C. The pH is between 3.4 and '
                '4.2. Both are measured on a composite sample from each delivered lot.'
            ),
            (
                'Microbiology: total plate count below 1000 CFU/g, yeast and mould below 100 '
                'CFU/g. Storage is ambient, below 25 °C, with a shelf life of 12 months.'
            ),
        ),
    ),
    CorpusDocument(
        doc_id='SPEC-PKG-021-R1',
        doc_code='SPEC-PKG-021',
        title='Specification: PET bottle 500 ml',
        doc_type=DocType.SPEC,
        revision=1,
        status=DocStatus.CURRENT,
        effective_date=date(2026, 1, 5),
        product_code='PKG-021',
        passages=_passages(
            'SPEC-PKG-021-R1',
            (
                'PET bottle, 500 ml nominal, clear, suitable for cold fill and high pressure '
                'processing. Mass is 24 g with a tolerance of ± 1 g. The neck finish is 38 mm to '
                'match the tamper-evident cap.'
            ),
            (
                'Brimful capacity is 520 ml ± 5 ml. Wall thickness is at least 0.25 mm at the '
                'label panel. Bottles must withstand HPP at 600 MPa without permanent '
                'deformation greater than 2 mm.'
            ),
            (
                'Acceptance: incoming lots are inspected to AQL 1.0 at general inspection level '
                'II. Critical defects such as cracks or open necks have an acceptance number of '
                'zero.'
            ),
        ),
    ),
    CorpusDocument(
        doc_id='SPEC-ING-005-R1',
        doc_code='SPEC-ING-005',
        title='Specification: ginger root',
        doc_type=DocType.SPEC,
        revision=1,
        status=DocStatus.CURRENT,
        effective_date=date(2026, 1, 20),
        product_code='ING-005',
        passages=_passages(
            'SPEC-ING-005-R1',
            (
                'Fresh ginger root, mature and firm, free from mould, sprouting and soil. Cleaned '
                'root carries no more than 2 % adhering soil by mass.'
            ),
            (
                'Moisture is 85 % to 90 %. Total plate count is below 100000 CFU/g before '
                'washing. Pesticide residues must be below the applicable maximum residue limits.'
            ),
            (
                'Delivery: ventilated crates of 20 kg, each labelled with the supplier lot '
                'number, the harvest date and the farm code. Crates without a label are rejected.'
            ),
        ),
    ),
)

DEMO_QUESTIONS: tuple[CorpusQuestion, ...] = (
    # Single-passage questions whose answer is the same in every revision of the document.
    CorpusQuestion(
        question_id='Q-S01',
        kind=QuestionKind.SINGLE,
        question='Within what time must a non-conformance record be opened after detection?',
        expected_answer='Within 24 hours of detection, by the person who found the deviation.',
        source_passage_ids=('SOP-QA-031-R1#p1',),
        required_facts=('within 24 hours',),
    ),
    CorpusQuestion(
        question_id='Q-S02',
        kind=QuestionKind.SINGLE,
        question='Which root cause method is used for major or repeat events?',
        expected_answer='A fishbone diagram.',
        source_passage_ids=('SOP-QA-031-R1#p2',),
        required_facts=('fishbone diagram',),
    ),
    CorpusQuestion(
        question_id='Q-S03',
        kind=QuestionKind.SINGLE,
        question='When is the CAPA effectiveness check performed and by whom?',
        expected_answer='90 days after closure, by the QA lead.',
        source_passage_ids=('SOP-QA-031-R1#p3',),
        required_facts=('90 days after closure', 'QA lead'),
    ),
    CorpusQuestion(
        question_id='Q-S04',
        kind=QuestionKind.SINGLE,
        question='How often is the hydraulic oil filter of the cold press replaced?',
        expected_answer='Every 500 operating hours.',
        source_passage_ids=('SOP-MT-008-R1#p2',),
        required_facts=('500 operating hours',),
    ),
    CorpusQuestion(
        question_id='Q-S05',
        kind=QuestionKind.SINGLE,
        question='How often are the cold press pressure gauges calibrated?',
        expected_answer='Every 6 months, against a reference gauge.',
        source_passage_ids=('SOP-MT-008-R1#p3',),
        required_facts=('every 6 months',),
    ),
    CorpusQuestion(
        question_id='Q-S06',
        kind=QuestionKind.SINGLE,
        question='What is the target time to complete a full mock-recall trace?',
        expected_answer='Within 4 hours.',
        source_passage_ids=('SOP-QA-040-R1#p2',),
        required_facts=('within 4 hours',),
    ),
    CorpusQuestion(
        question_id='Q-S07',
        kind=QuestionKind.SINGLE,
        question='What mass-balance tolerance applies in a mock recall?',
        expected_answer='Produced, shipped and on-hand quantity must reconcile within 2 %.',
        source_passage_ids=('SOP-QA-040-R1#p2',),
        required_facts=('within 2 %',),
    ),
    CorpusQuestion(
        question_id='Q-S08',
        kind=QuestionKind.SINGLE,
        question='What are the mass and mass tolerance of the 500 ml PET bottle?',
        expected_answer='24 g with a tolerance of ± 1 g.',
        source_passage_ids=('SPEC-PKG-021-R1#p1',),
        required_facts=('24 g', '± 1 g'),
    ),
    CorpusQuestion(
        question_id='Q-S09',
        kind=QuestionKind.SINGLE,
        question='Which AQL applies to incoming PET bottle lots?',
        expected_answer='AQL 1.0 at general inspection level II.',
        source_passage_ids=('SPEC-PKG-021-R1#p3',),
        required_facts=('AQL 1.0',),
    ),
    CorpusQuestion(
        question_id='Q-S10',
        kind=QuestionKind.SINGLE,
        question='What ferrous test piece size verifies the metal detector?',
        expected_answer='A 2.0 mm ferrous test piece.',
        source_passage_ids=(f'{_HACCP_R2}#p4',),
        required_facts=('2.0 mm ferrous',),
    ),
    CorpusQuestion(
        question_id='Q-S11',
        kind=QuestionKind.SINGLE,
        question='How is ginger root delivered and what must each crate label show?',
        expected_answer=(
            'In ventilated crates of 20 kg labelled with supplier lot number, harvest date '
            'and farm code.'
        ),
        source_passage_ids=('SPEC-ING-005-R1#p3',),
        required_facts=('20 kg', 'supplier lot number'),
    ),
    CorpusQuestion(
        question_id='Q-S12',
        kind=QuestionKind.SINGLE,
        question='Above what core temperature is a produce delivery always rejected?',
        expected_answer='Above 10 °C.',
        source_passage_ids=(f'{_RECEIVING_R2}#p3',),
        required_facts=('above 10 °C',),
    ),
    CorpusQuestion(
        question_id='Q-S13',
        kind=QuestionKind.SINGLE,
        question='What moisture range does the ginger root specification allow?',
        expected_answer='85 % to 90 %.',
        source_passage_ids=('SPEC-ING-005-R1#p2',),
        required_facts=('85 % to 90 %',),
    ),
    # Multi-hop questions: the answer needs passages from two or more documents.
    CorpusQuestion(
        question_id='Q-M01',
        kind=QuestionKind.MULTI_HOP,
        question=(
            'If the metal detector fails verification, what happens to the product, and within '
            'what time must the non-conformance record be opened?'
        ),
        expected_answer=(
            'Product packed since the last passed verification is placed on hold and '
            're-inspected; the NCR is opened within 24 hours of detection.'
        ),
        source_passage_ids=(f'{_HACCP_R2}#p5', 'SOP-QA-031-R1#p1'),
        required_facts=('placed on hold', 'within 24 hours'),
    ),
    CorpusQuestion(
        question_id='Q-M02',
        kind=QuestionKind.MULTI_HOP,
        question=(
            'Can the 500 ml PET bottle withstand the minimum pressure set in the current HACCP '
            'plan?'
        ),
        expected_answer=(
            'Yes. The bottle must withstand HPP at 600 MPa, which equals the current minimum '
            'critical limit of 600 MPa.'
        ),
        source_passage_ids=('SPEC-PKG-021-R1#p2', f'{_HACCP_R2}#p2'),
        required_facts=('withstand HPP at 600 MPa', 'minimum pressure of 600 MPa'),
    ),
    CorpusQuestion(
        question_id='Q-M03',
        kind=QuestionKind.MULTI_HOP,
        question=(
            'A ginger delivery arrives at 8 °C. May it be released, by whom, and what lot '
            'information must its crates carry?'
        ),
        expected_answer=(
            'It is above 6 °C, so it is held; between 6 °C and 10 °C the QA lead may release it '
            'after a documented risk assessment. Crates carry the supplier lot number, harvest '
            'date and farm code.'
        ),
        source_passage_ids=(f'{_RECEIVING_R2}#p3', 'SPEC-ING-005-R1#p3'),
        required_facts=('between 6 °C and 10 °C', 'QA lead', 'supplier lot number'),
    ),
    CorpusQuestion(
        question_id='Q-M04',
        kind=QuestionKind.MULTI_HOP,
        question=(
            'What ATP limit applies after CIP, what happens if the swab fails, and how soon must '
            'the deviation be recorded as a non-conformance?'
        ),
        expected_answer=(
            'The result must be 150 RLU or lower; otherwise the CIP cycle is repeated. The NCR '
            'is opened within 24 hours.'
        ),
        source_passage_ids=(f'{_CIP_R2}#p3', 'SOP-QA-031-R1#p1'),
        required_facts=('150 RLU or lower', 'is repeated', 'within 24 hours'),
    ),
    # Superseded traps: the old revision answers differently. Citing it is a citation error.
    CorpusQuestion(
        question_id='Q-T01',
        kind=QuestionKind.SUPERSEDED,
        question='What is the maximum core temperature for accepting a chilled produce delivery?',
        expected_answer='6 °C or lower.',
        source_passage_ids=(f'{_RECEIVING_R2}#p2',),
        required_facts=('6 °C or lower',),
        stale_passage_ids=(f'{_RECEIVING_R1}#p2',),
    ),
    CorpusQuestion(
        question_id='Q-T02',
        kind=QuestionKind.SUPERSEDED,
        question='What caustic concentration, temperature and time apply to CIP of the line?',
        expected_answer='A 2.0 % caustic soda solution at 75 °C for 25 minutes.',
        source_passage_ids=(f'{_CIP_R2}#p2',),
        required_facts=('2.0 %', '75 °C', '25 minutes'),
        stale_passage_ids=(f'{_CIP_R1}#p2',),
    ),
    CorpusQuestion(
        question_id='Q-T03',
        kind=QuestionKind.SUPERSEDED,
        question='What is the critical limit for pressure at CCP 1, high pressure processing?',
        expected_answer='A minimum of 600 MPa held for 180 seconds.',
        source_passage_ids=(f'{_HACCP_R2}#p2',),
        required_facts=('600 MPa',),
        stale_passage_ids=(f'{_HACCP_R1}#p2',),
    ),
    CorpusQuestion(
        question_id='Q-T04',
        kind=QuestionKind.SUPERSEDED,
        question='What is the minimum soluble solids level for mango puree?',
        expected_answer='15 °Brix at 20 °C.',
        source_passage_ids=(f'{_MANGO_R2}#p2',),
        required_facts=('15 °Brix',),
        stale_passage_ids=(f'{_MANGO_R1}#p2',),
    ),
    # Unanswerable questions: the correct behaviour is to abstain.
    CorpusQuestion(
        question_id='Q-U01',
        kind=QuestionKind.UNANSWERABLE,
        question='What is the maximum permitted patulin level in apple juice?',
        expected_answer=ABSTENTION_ANSWER,
    ),
    CorpusQuestion(
        question_id='Q-U02',
        kind=QuestionKind.UNANSWERABLE,
        question='What is the shelf life of the finished Green Detox juice?',
        expected_answer=ABSTENTION_ANSWER,
    ),
    CorpusQuestion(
        question_id='Q-U03',
        kind=QuestionKind.UNANSWERABLE,
        question='Which supplier is approved to deliver ginger root?',
        expected_answer=ABSTENTION_ANSWER,
    ),
    CorpusQuestion(
        question_id='Q-U04',
        kind=QuestionKind.UNANSWERABLE,
        question='How often is the allergen cleaning validation repeated?',
        expected_answer=ABSTENTION_ANSWER,
    ),
    CorpusQuestion(
        question_id='Q-U05',
        kind=QuestionKind.UNANSWERABLE,
        question='Which model of HPP vessel is installed on the line?',
        expected_answer=ABSTENTION_ANSWER,
    ),
)
