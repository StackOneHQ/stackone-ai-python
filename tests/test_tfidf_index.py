"""Tests for TF-IDF index implementation"""

import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from stackone_ai.utils.tfidf_index import TfidfDocument, TfidfIndex, tokenize

# Hypothesis strategies for PBT
# Text with various punctuation patterns
punctuation_text_strategy = st.text(
    alphabet=string.ascii_letters + string.punctuation + " ",
    min_size=1,
    max_size=100,
).filter(lambda s: any(c in string.ascii_letters for c in s))  # Must have some letters

# Common English stopwords (subset for testing)
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "is",
    "are",
    "was",
}

# Text containing stopwords mixed with content words
stopword_text_strategy = st.lists(
    st.one_of(
        st.sampled_from(list(STOPWORDS)),
        st.text(alphabet="abcdefghij", min_size=3, max_size=10).filter(lambda s: s.lower() not in STOPWORDS),
    ),
    min_size=1,
    max_size=20,
).map(lambda words: " ".join(words))

# Tool name patterns with underscores
tool_name_strategy = st.lists(
    st.text(alphabet=string.ascii_lowercase, min_size=2, max_size=10),
    min_size=1,
    max_size=4,
).map(lambda parts: "_".join(parts))

# Document collection for TF-IDF testing
document_strategy = st.builds(
    TfidfDocument,
    id=st.text(alphabet=string.ascii_lowercase + string.digits, min_size=1, max_size=20),
    text=st.text(alphabet=string.ascii_letters + " _", min_size=1, max_size=100),
)

# Query strings for search testing
query_strategy = st.text(
    alphabet=string.ascii_letters + " ",
    min_size=1,
    max_size=50,
).filter(lambda s: s.strip())


class TestTokenize:
    """Test tokenization functionality"""

    def test_basic_tokenization(self):
        """Test basic text tokenization"""
        text = "Hello World"
        tokens = tokenize(text)
        assert tokens == ["hello", "world"]

    def test_lowercase_conversion(self):
        """Test that text is lowercased"""
        text = "UPPERCASE lowercase MiXeD"
        tokens = tokenize(text)
        assert all(t.islower() for t in tokens)

    def test_punctuation_removal(self):
        """Test that punctuation is removed"""
        text = "Hello, world! How are you?"
        tokens = tokenize(text)
        assert "," not in tokens
        assert "!" not in tokens
        assert "?" not in tokens

    def test_stopword_filtering(self):
        """Test that stopwords are removed"""
        text = "the quick brown fox and the lazy dog"
        tokens = tokenize(text)
        # Stopwords should be filtered
        assert "the" not in tokens
        assert "and" not in tokens
        # Content words should remain
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens
        assert "lazy" in tokens
        assert "dog" in tokens

    def test_underscore_preservation(self):
        """Test that underscores are preserved"""
        text = "hibob_list_employees"
        tokens = tokenize(text)
        assert "hibob_list_employees" in tokens

    def test_empty_string(self):
        """Test tokenization of empty string"""
        tokens = tokenize("")
        assert tokens == []

    def test_only_stopwords(self):
        """Test text with only stopwords"""
        text = "the a an and or but"
        tokens = tokenize(text)
        assert tokens == []

    @given(text=punctuation_text_strategy)
    @settings(max_examples=100)
    def test_punctuation_removal_pbt(self, text: str):
        """PBT: Test that all punctuation is removed from tokens."""
        tokens = tokenize(text)
        # No token should contain punctuation (except underscore which is preserved)
        for token in tokens:
            non_underscore_punct = set(string.punctuation) - {"_"}
            assert not any(c in non_underscore_punct for c in token), f"Token '{token}' contains punctuation"

    @given(text=punctuation_text_strategy)
    @settings(max_examples=100)
    def test_lowercase_conversion_pbt(self, text: str):
        """PBT: Test that all tokens are lowercase."""
        tokens = tokenize(text)
        assert all(t.islower() or "_" in t for t in tokens), f"Not all tokens are lowercase: {tokens}"

    @given(tool_name=tool_name_strategy)
    @settings(max_examples=100)
    def test_underscore_preservation_pbt(self, tool_name: str):
        """PBT: Test that underscores in tool names are preserved."""
        tokens = tokenize(tool_name)
        # If the tool name contains underscores and is long enough, underscores should be preserved
        if "_" in tool_name and len(tool_name) > 2:
            # Either the full tool name is preserved, or we get tokens with underscores
            # Note: very short parts may be filtered as stopwords
            assert tool_name.lower() in tokens or any("_" in t for t in tokens) or len(tokens) == 0

    @given(text=stopword_text_strategy)
    @settings(max_examples=100)
    def test_stopword_filtering_pbt(self, text: str):
        """PBT: Test that stopwords are filtered out."""
        tokens = tokenize(text)
        # No stopword should appear in the result
        for token in tokens:
            assert token.lower() not in STOPWORDS, f"Stopword '{token}' was not filtered"


class TestTfidfIndex:
    """Test TF-IDF index functionality"""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            TfidfDocument(id="doc1", text="create new employee in hris system"),
            TfidfDocument(id="doc2", text="list all employees from database"),
            TfidfDocument(id="doc3", text="update employee information"),
            TfidfDocument(id="doc4", text="delete employee record"),
            TfidfDocument(id="doc5", text="search for candidates in ats"),
            TfidfDocument(id="doc6", text="create job posting"),
        ]

    def test_index_creation(self, sample_documents):
        """Test that index can be created"""
        index = TfidfIndex()
        index.build(sample_documents)

        assert len(index.vocab) > 0
        assert len(index.idf) == len(index.vocab)
        assert len(index.docs) == len(sample_documents)

    def test_vocabulary_building(self, sample_documents):
        """Test vocabulary is built correctly"""
        index = TfidfIndex()
        index.build(sample_documents)

        # Check that content words are in vocabulary
        assert any("employee" in term for term in index.vocab.keys())
        assert any("create" in term for term in index.vocab.keys())
        assert any("hris" in term for term in index.vocab.keys())

    def test_search_returns_results(self, sample_documents):
        """Test that search returns relevant results"""
        index = TfidfIndex()
        index.build(sample_documents)

        results = index.search("employee", k=5)

        assert len(results) > 0
        # Results should be sorted by score
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_search_relevance(self, sample_documents):
        """Test that search returns relevant documents"""
        index = TfidfIndex()
        index.build(sample_documents)

        # Search for "employee"
        results = index.search("employee", k=5)

        # Top results should contain employee-related docs
        top_ids = {r.id for r in results[:3]}
        assert "doc1" in top_ids or "doc2" in top_ids or "doc3" in top_ids

    def test_search_with_multiple_terms(self, sample_documents):
        """Test search with multiple query terms"""
        index = TfidfIndex()
        index.build(sample_documents)

        results = index.search("create employee hris", k=5)

        assert len(results) > 0
        # doc1 should be highly ranked (contains all three terms)
        top_ids = [r.id for r in results[:2]]
        assert "doc1" in top_ids

    def test_search_limit(self, sample_documents):
        """Test that search respects k parameter"""
        index = TfidfIndex()
        index.build(sample_documents)

        results = index.search("employee", k=2)
        assert len(results) <= 2

        results = index.search("employee", k=10)
        # Should return at most the number of documents
        assert len(results) <= len(sample_documents)

    def test_score_range(self, sample_documents):
        """Test that scores are in [0, 1] range"""
        index = TfidfIndex()
        index.build(sample_documents)

        results = index.search("employee", k=10)

        for result in results:
            assert 0.0 <= result.score <= 1.0

    @given(
        documents=st.lists(document_strategy, min_size=1, max_size=20),
        query=query_strategy,
        k=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=50)
    def test_score_range_pbt(self, documents: list[TfidfDocument], query: str, k: int):
        """PBT: Test that scores are always in [0, 1] range."""
        index = TfidfIndex()
        index.build(documents)

        results = index.search(query, k=k)

        # All scores must be in valid range
        for result in results:
            assert 0.0 <= result.score <= 1.0, f"Score {result.score} out of range"

    @given(
        documents=st.lists(document_strategy, min_size=1, max_size=20),
        query=query_strategy,
        k=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=50)
    def test_results_sorted_descending_pbt(self, documents: list[TfidfDocument], query: str, k: int):
        """PBT: Test that results are always sorted by score descending."""
        index = TfidfIndex()
        index.build(documents)

        results = index.search(query, k=k)

        # Results should be sorted by score in descending order
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score, (
                f"Results not sorted: {results[i].score} < {results[i + 1].score}"
            )

    @given(
        documents=st.lists(document_strategy, min_size=1, max_size=20),
        query=query_strategy,
        k=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=50)
    def test_result_count_respects_limit_pbt(self, documents: list[TfidfDocument], query: str, k: int):
        """PBT: Test that result count never exceeds k or document count."""
        index = TfidfIndex()
        index.build(documents)

        results = index.search(query, k=k)

        # Result count should not exceed k or total documents
        assert len(results) <= k
        assert len(results) <= len(documents)

    def test_empty_query(self, sample_documents):
        """Test search with empty query"""
        index = TfidfIndex()
        index.build(sample_documents)

        results = index.search("", k=5)
        assert results == []

    def test_no_matching_terms(self, sample_documents):
        """Test search with terms not in vocabulary"""
        index = TfidfIndex()
        index.build(sample_documents)

        results = index.search("xyzabc", k=5)
        assert results == []

    def test_stopword_query(self, sample_documents):
        """Test search with only stopwords"""
        index = TfidfIndex()
        index.build(sample_documents)

        results = index.search("the and or", k=5)
        assert results == []

    def test_empty_corpus(self):
        """Test building index with empty corpus"""
        index = TfidfIndex()
        index.build([])

        assert len(index.vocab) == 0
        assert len(index.docs) == 0

        results = index.search("test", k=5)
        assert results == []

    def test_single_document(self):
        """Test with single document"""
        index = TfidfIndex()
        docs = [TfidfDocument(id="doc1", text="single document test")]
        index.build(docs)

        results = index.search("document", k=5)
        assert len(results) == 1
        assert results[0].id == "doc1"
        assert results[0].score > 0

    def test_duplicate_documents(self):
        """Test with duplicate document IDs"""
        index = TfidfIndex()
        docs = [
            TfidfDocument(id="doc1", text="first document"),
            TfidfDocument(id="doc1", text="duplicate id"),
        ]
        index.build(docs)

        # Both documents should be in index
        assert len(index.docs) == 2

    def test_case_insensitive_search(self, sample_documents):
        """Test that search is case-insensitive"""
        index = TfidfIndex()
        index.build(sample_documents)

        results_lower = index.search("employee", k=5)
        results_upper = index.search("EMPLOYEE", k=5)
        results_mixed = index.search("EmPlOyEe", k=5)

        # Should return same results (same IDs in same order)
        assert len(results_lower) == len(results_upper) == len(results_mixed)
        assert [r.id for r in results_lower] == [r.id for r in results_upper]
        assert [r.id for r in results_lower] == [r.id for r in results_mixed]

    def test_special_characters_in_query(self, sample_documents):
        """Test search with special characters"""
        index = TfidfIndex()
        index.build(sample_documents)

        # Special characters should be stripped
        results = index.search("employee!", k=5)
        assert len(results) > 0

        results2 = index.search("employee", k=5)
        # Should return same results
        assert [r.id for r in results] == [r.id for r in results2]

    def test_idf_calculation(self):
        """Test IDF values are calculated correctly"""
        index = TfidfIndex()
        docs = [
            TfidfDocument(id="doc1", text="common word appears everywhere"),
            TfidfDocument(id="doc2", text="common word appears here too"),
            TfidfDocument(id="doc3", text="common word and rare term"),
        ]
        index.build(docs)

        # "common" appears in all docs, should have lower IDF
        # "rare" appears in one doc, should have higher IDF
        common_id = index.vocab.get("common")
        rare_id = index.vocab.get("rare")

        if common_id is not None and rare_id is not None:
            assert index.idf[rare_id] > index.idf[common_id]


class TestTfidfDocument:
    """Test TfidfDocument named tuple"""

    def test_document_creation(self):
        """Test creating a document"""
        doc = TfidfDocument(id="test", text="test text")
        assert doc.id == "test"
        assert doc.text == "test text"

    def test_document_immutability(self):
        """Test that TfidfDocument is immutable"""
        doc = TfidfDocument(id="test", text="test text")
        with pytest.raises(AttributeError):
            doc.id = "new_id"  # type: ignore


class TestTfidfIntegration:
    """Integration tests for TF-IDF with realistic scenarios"""

    def test_tool_name_matching(self):
        """Test matching tool names"""
        index = TfidfIndex()
        docs = [
            TfidfDocument(id="hibob_create_employee", text="create employee hibob system"),
            TfidfDocument(id="hibob_list_employees", text="list employees hibob system"),
            TfidfDocument(id="bamboohr_create_candidate", text="create candidate bamboohr system"),
            TfidfDocument(id="workday_list_contacts", text="list contacts workday system"),
        ]
        index.build(docs)

        # Search for HiBob tools
        results = index.search("employee hibob", k=5)
        top_ids = [r.id for r in results[:2]]
        assert "hibob_create_employee" in top_ids or "hibob_list_employees" in top_ids

        # Search for create operations
        results = index.search("create", k=5)
        assert len(results) > 0
        # Should find multiple create tools
        create_count = sum(1 for r in results if "create" in r.id)
        assert create_count >= 2
