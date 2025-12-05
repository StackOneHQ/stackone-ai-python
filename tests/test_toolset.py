"""Tests for StackOneToolSet."""

from stackone_ai.toolset import StackOneToolSet


def test_set_accounts():
    """Test setting account IDs for filtering"""
    toolset = StackOneToolSet(api_key="test_key")
    result = toolset.set_accounts(["acc1", "acc2"])

    # Should return self for chaining
    assert result is toolset
    assert toolset._account_ids == ["acc1", "acc2"]


def test_filter_by_provider():
    """Test provider filtering"""
    toolset = StackOneToolSet(api_key="test_key")

    # Test matching providers
    assert toolset._filter_by_provider("hris_list_employees", ["hris", "ats"])
    assert toolset._filter_by_provider("ats_create_job", ["hris", "ats"])

    # Test non-matching providers
    assert not toolset._filter_by_provider("crm_list_contacts", ["hris", "ats"])

    # Test case-insensitive matching
    assert toolset._filter_by_provider("HRIS_list_employees", ["hris"])
    assert toolset._filter_by_provider("hris_list_employees", ["HRIS"])


def test_filter_by_action():
    """Test action filtering with glob patterns"""
    toolset = StackOneToolSet(api_key="test_key")

    # Test exact match
    assert toolset._filter_by_action("hris_list_employees", ["hris_list_employees"])

    # Test glob pattern
    assert toolset._filter_by_action("hris_list_employees", ["*_list_employees"])
    assert toolset._filter_by_action("ats_list_employees", ["*_list_employees"])
    assert toolset._filter_by_action("hris_list_employees", ["hris_*"])
    assert toolset._filter_by_action("hris_create_employee", ["hris_*"])

    # Test non-matching patterns
    assert not toolset._filter_by_action("crm_list_contacts", ["*_list_employees"])
    assert not toolset._filter_by_action("ats_create_job", ["hris_*"])


def test_matches_filter_positive_patterns():
    """Test _matches_filter with positive patterns"""
    toolset = StackOneToolSet(api_key="test_key")

    # Single pattern
    assert toolset._matches_filter("hris_list_employees", "hris_*")
    assert toolset._matches_filter("ats_create_job", "ats_*")
    assert not toolset._matches_filter("crm_contacts", "hris_*")

    # Multiple patterns (OR logic)
    assert toolset._matches_filter("hris_list_employees", ["hris_*", "ats_*"])
    assert toolset._matches_filter("ats_create_job", ["hris_*", "ats_*"])
    assert not toolset._matches_filter("crm_contacts", ["hris_*", "ats_*"])


def test_matches_filter_negative_patterns():
    """Test _matches_filter with negative patterns (exclusion)"""
    toolset = StackOneToolSet(api_key="test_key")

    # Negative pattern
    assert not toolset._matches_filter("hris_delete_employee", ["hris_*", "!hris_delete_*"])
    assert toolset._matches_filter("hris_list_employees", ["hris_*", "!hris_delete_*"])

    # Only negative patterns (should match everything not excluded)
    assert not toolset._matches_filter("hris_delete_employee", "!hris_delete_*")
    assert toolset._matches_filter("hris_list_employees", "!hris_delete_*")
