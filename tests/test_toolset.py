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
    assert toolset._filter_by_provider("hibob_list_employees", ["hibob", "bamboohr"])
    assert toolset._filter_by_provider("bamboohr_create_job", ["hibob", "bamboohr"])

    # Test non-matching providers
    assert not toolset._filter_by_provider("workday_list_contacts", ["hibob", "bamboohr"])

    # Test case-insensitive matching
    assert toolset._filter_by_provider("HIBOB_list_employees", ["hibob"])
    assert toolset._filter_by_provider("hibob_list_employees", ["HIBOB"])


def test_filter_by_action():
    """Test action filtering with glob patterns"""
    toolset = StackOneToolSet(api_key="test_key")

    # Test exact match
    assert toolset._filter_by_action("hibob_list_employees", ["hibob_list_employees"])

    # Test glob pattern
    assert toolset._filter_by_action("hibob_list_employees", ["*_list_employees"])
    assert toolset._filter_by_action("bamboohr_list_employees", ["*_list_employees"])
    assert toolset._filter_by_action("hibob_list_employees", ["hibob_*"])
    assert toolset._filter_by_action("hibob_create_employee", ["hibob_*"])

    # Test non-matching patterns
    assert not toolset._filter_by_action("workday_list_contacts", ["*_list_employees"])
    assert not toolset._filter_by_action("bamboohr_create_job", ["hibob_*"])
