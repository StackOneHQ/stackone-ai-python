"""Tests for TF-IDF index implementation"""

import pytest

from stackone_ai.utils.tfidf_index import TfidfDocument, TfidfIndex, tokenize


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
        text = "hris_list_employees"
        tokens = tokenize(text)
        assert "hris_list_employees" in tokens

    def test_empty_string(self):
        """Test tokenization of empty string"""
        tokens = tokenize("")
        assert tokens == []

    def test_only_stopwords(self):
        """Test text with only stopwords"""
        text = "the a an and or but"
        tokens = tokenize(text)
        assert tokens == []


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
            TfidfDocument(id="hris_create_employee", text="create employee hris system"),
            TfidfDocument(id="hris_list_employees", text="list employees hris system"),
            TfidfDocument(id="ats_create_candidate", text="create candidate ats system"),
            TfidfDocument(id="crm_list_contacts", text="list contacts crm system"),
        ]
        index.build(docs)

        # Search for HRIS tools
        results = index.search("employee hris", k=5)
        top_ids = [r.id for r in results[:2]]
        assert "hris_create_employee" in top_ids or "hris_list_employees" in top_ids

        # Search for create operations
        results = index.search("create", k=5)
        assert len(results) > 0
        # Should find multiple create tools
        create_count = sum(1 for r in results if "create" in r.id)
        assert create_count >= 2
