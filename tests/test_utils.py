"""Unit tests for utils.py high-priority functions."""
import pytest

from utils import (
    get_questionnaire,
    convert_data,
    compute_statistics,
    hypothesis_testing,
    parsing,
)


class TestGetQuestionnaire:
    """Tests for get_questionnaire."""

    def test_valid_name_returns_questionnaire(self):
        """Calling with existing name returns dict with expected keys."""
        q = get_questionnaire("Empathy")
        assert q["name"] == "Empathy"
        assert "questions" in q
        assert "scale" in q
        assert "reverse" in q
        assert "categories" in q

    def test_unknown_name_raises_value_error(self):
        """Calling with non-existent name raises ValueError."""
        with pytest.raises(ValueError, match="Questionnaire not found"):
            get_questionnaire("NonExistent")

    def test_missing_file_raises_file_not_found(self, tmp_path, monkeypatch):
        """When questionnaires.json does not exist, FileNotFoundError is raised."""
        monkeypatch.chdir(tmp_path)
        assert not (tmp_path / "questionnaires.json").exists()
        with pytest.raises(FileNotFoundError) as exc_info:
            get_questionnaire("X")
        assert "questionnaires.json" in str(exc_info.value)


class TestConvertData:
    """Tests for convert_data."""

    def test_reverse_scaling(
        self, questionnaire_scale8_reverse1, csv_one_order_one_test
    ):
        """Reverse-scale items are transformed with scale - value."""
        test_data = convert_data(
            questionnaire_scale8_reverse1, str(csv_one_order_one_test)
        )
        assert len(test_data) == 1
        column_data = test_data[0]
        assert column_data[1] == 3  # 8 - 5
        assert column_data[2] == 4
        assert column_data[3] == 4

    def test_no_reverse(self, questionnaire_no_reverse, csv_no_reverse):
        """Without reverse list, values are stored as-is."""
        test_data = convert_data(
            questionnaire_no_reverse, str(csv_no_reverse)
        )
        assert len(test_data) == 1
        assert test_data[0] == {1: 2, 2: 3, 3: 4}

    def test_multiple_orders(
        self, questionnaire_scale8_reverse1, csv_two_orders
    ):
        """CSV with two order blocks yields two column_data dicts."""
        test_data = convert_data(
            questionnaire_scale8_reverse1, str(csv_two_orders)
        )
        assert len(test_data) == 2
        # First block: 1->8-5=3, 2->4, 3->4
        assert test_data[0] == {1: 3, 2: 4, 3: 4}
        # Second block: order 2,3,1 with responses 2,3,1 -> 1 is reverse: 8-1=7
        assert test_data[1] == {2: 2, 3: 3, 1: 7}

    def test_missing_file_raises_system_exit(self, questionnaire_avg, tmp_path):
        """Calling with non-existent path exits with code 1."""
        missing = tmp_path / "nonexistent.csv"
        assert not missing.exists()
        with pytest.raises(SystemExit) as exc_info:
            convert_data(questionnaire_avg, str(missing))
        assert exc_info.value.code == 1

    def test_invalid_cell_raises_system_exit(
        self, questionnaire_scale8_reverse1, csv_invalid_cell
    ):
        """Non-integer in response cell causes sys.exit(1)."""
        with pytest.raises(SystemExit) as exc_info:
            convert_data(
                questionnaire_scale8_reverse1, str(csv_invalid_cell)
            )
        assert exc_info.value.code == 1


class TestComputeStatistics:
    """Tests for compute_statistics."""

    def test_avg_mode(self, questionnaire_avg):
        """AVG mode: per-case average then mean/stdev across cases."""
        data_list = [
            {1: 2, 2: 4, 3: 6},  # avg 4
            {1: 4, 2: 4, 3: 4},  # avg 4
        ]
        results = compute_statistics(questionnaire_avg, data_list)
        assert len(results) == 1
        mean_val, std_val, n = results[0]
        assert mean_val == 4.0
        assert std_val == 0.0
        assert n == 2

    def test_sum_mode(self, questionnaire_sum):
        """SUM mode: per-case sum then mean/stdev across cases."""
        data_list = [
            {1: 2, 2: 4, 3: 6},  # sum 12
            {1: 4, 2: 4, 3: 4},  # sum 12
        ]
        results = compute_statistics(questionnaire_sum, data_list)
        assert len(results) == 1
        mean_val, std_val, n = results[0]
        assert mean_val == 12.0
        assert std_val == 0.0
        assert n == 2

    def test_two_categories(self, questionnaire_two_categories):
        """Two categories produce two result tuples."""
        data_list = [
            {1: 2, 2: 4, 3: 1},
            {1: 4, 2: 6, 3: 3},
        ]
        results = compute_statistics(
            questionnaire_two_categories, data_list
        )
        assert len(results) == 2
        # Cat A: [1,2] -> (2+4)/2=3 and (4+6)/2=5 -> mean 4, stdev ~1.41
        assert results[0][0] == 4.0
        assert results[0][2] == 2
        # Cat B: [3] -> 1 and 3 -> mean 2, stdev ~1.41
        assert results[1][0] == 2.0
        assert results[1][2] == 2

    def test_fewer_than_two_cases_raises(self, questionnaire_avg):
        """Fewer than 2 test cases raises ValueError."""
        data_list = [{1: 1, 2: 2, 3: 3}]
        with pytest.raises(ValueError, match="at least 2 test cases"):
            compute_statistics(questionnaire_avg, data_list)


class TestHypothesisTesting:
    """Tests for hypothesis_testing."""

    def test_equal_means_equal_var_cannot_reject(self):
        """Equal means and variance: H0 cannot be rejected."""
        result1 = (10.0, 1.0, 100)
        result2 = (10.0, 1.0, 100)
        output_text, output_list = hypothesis_testing(
            result1, result2, 0.05, "model", "Crowd"
        )
        assert "cannot be rejected" in output_text
        assert output_list.endswith(" | ")

    def test_unequal_variance_welch_path(self):
        """Very different variances trigger Welch / unequal conclusion."""
        result1 = (10.0, 0.1, 10)
        result2 = (10.0, 5.0, 10)
        output_text, _ = hypothesis_testing(
            result1, result2, 0.05, "model", "Crowd"
        )
        assert "Welch" in output_text or "unequal" in output_text.lower()

    def test_mean1_greater_than_mean2(self):
        """When model mean > crowd mean, text says 'larger than'."""
        result1 = (20.0, 1.0, 100)
        result2 = (10.0, 1.0, 100)
        output_text, _ = hypothesis_testing(
            result1, result2, 0.05, "model", "Crowd"
        )
        assert "larger than" in output_text

    def test_mean1_less_than_mean2(self):
        """When model mean < crowd mean, text says 'smaller than'."""
        result1 = (5.0, 1.0, 100)
        result2 = (10.0, 1.0, 100)
        output_text, _ = hypothesis_testing(
            result1, result2, 0.05, "model", "Crowd"
        )
        assert "smaller than" in output_text

    def test_t_value_direction(self):
        """Higher mean1 yields positive t-value (sanity check)."""
        result1 = (15.0, 1.0, 50)
        result2 = (10.0, 1.0, 50)
        output_text, _ = hypothesis_testing(
            result1, result2, 0.05, "model", "Crowd"
        )
        assert "t-value" in output_text
        # t should be positive; we only check that we get a conclusion
        assert "larger than" in output_text


class TestParsing:
    """Tests for parsing (16 personalities score list -> code and role)."""

    def test_boundary_50(self):
        """Scores exactly 50 yield E, N, T, J, -A (>= 50 is Assertive)."""
        code, role = parsing([50, 50, 50, 50, 50])
        assert code == "ENTJ-A"
        assert role == "Commander"

    def test_just_below_50(self):
        """Scores 49 yield I, S, F, P, -T."""
        code, role = parsing([49, 49, 49, 49, 49])
        assert code == "ISFP-T"
        assert role == "Adventurer"

    def test_all_high(self):
        """All high scores yield ENTJ-A."""
        code, role = parsing([100, 100, 100, 100, 100])
        assert code == "ENTJ-A"
        assert role == "Commander"

    def test_all_low(self):
        """All low scores yield ISFP-T."""
        code, role = parsing([0, 0, 0, 0, 0])
        assert code == "ISFP-T"
        assert role == "Adventurer"

    def test_infp_type(self):
        """INFP: I, N, F, P."""
        code, role = parsing([49, 50, 49, 49, 50])
        assert code[:4] == "INFP"
        assert role == "Mediator"

    def test_estj_type(self):
        """ESTJ: E, S, T, J."""
        code, role = parsing([50, 49, 50, 50, 49])
        assert code[:4] == "ESTJ"
        assert role == "Executive"
