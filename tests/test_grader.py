"""Unit tests for the grading system."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openenv_submission.grader import (
    GradeResult,
    MIN_SCORE,
    MAX_SCORE,
    clamp_score,
    grade_diagnosis,
    grade,
    grade_trajectory,
    time_decay_factor,
    grade_easy_fever_cough,
    grade_medium_flu_vs_dengue,
    grade_hard_dengue_like,
    grade_expert_malaria_mimic,
    grade_expert_typhoid_enteric,
    grade_medium_pneumonia,
    grade_hard_covid_respiratory,
    grade_expert_chikungunya,
    TASK_GRADERS,
)
from openenv_submission.tasks import CASES, CASE_BY_NAME, list_task_names


# ---------------------------------------------------------------------------
# clamp_score
# ---------------------------------------------------------------------------

def test_clamp_score_zero():
    assert clamp_score(0.0) == MIN_SCORE

def test_clamp_score_one():
    assert clamp_score(1.0) == MAX_SCORE

def test_clamp_score_negative():
    assert clamp_score(-5.0) == MIN_SCORE

def test_clamp_score_above_one():
    assert clamp_score(2.5) == MAX_SCORE

def test_clamp_score_in_range():
    assert clamp_score(0.5) == 0.5

def test_clamp_score_strictly_in_open_interval():
    s = clamp_score(0.5)
    assert 0.0 < s < 1.0


# ---------------------------------------------------------------------------
# grade_diagnosis — exact matches
# ---------------------------------------------------------------------------

def test_exact_match_common_cold():
    r = grade_diagnosis("common cold", "common cold")
    assert r.is_correct and r.score == MAX_SCORE

def test_exact_match_dengue():
    r = grade_diagnosis("dengue", "dengue")
    assert r.is_correct and r.score == MAX_SCORE

def test_exact_match_malaria():
    r = grade_diagnosis("malaria", "malaria")
    assert r.is_correct

def test_exact_match_typhoid():
    r = grade_diagnosis("typhoid", "typhoid")
    assert r.is_correct

def test_exact_match_pneumonia():
    r = grade_diagnosis("pneumonia", "pneumonia")
    assert r.is_correct

def test_exact_match_covid():
    r = grade_diagnosis("covid-19", "covid-19")
    assert r.is_correct

def test_exact_match_chikungunya():
    r = grade_diagnosis("chikungunya", "chikungunya")
    assert r.is_correct


# ---------------------------------------------------------------------------
# grade_diagnosis — synonym matches
# ---------------------------------------------------------------------------

def test_synonym_flu():
    r = grade_diagnosis("flu", "influenza")
    assert r.is_correct and r.score == MAX_SCORE

def test_synonym_cold():
    r = grade_diagnosis("cold", "common cold")
    assert r.is_correct

def test_synonym_dengue_fever():
    r = grade_diagnosis("dengue fever", "dengue")
    assert r.is_correct

def test_synonym_enteric_fever():
    r = grade_diagnosis("enteric fever", "typhoid")
    assert r.is_correct

def test_synonym_covid():
    r = grade_diagnosis("covid", "covid-19")
    assert r.is_correct

def test_synonym_corona():
    r = grade_diagnosis("corona", "covid-19")
    assert r.is_correct

def test_synonym_cap():
    r = grade_diagnosis("cap", "pneumonia")
    assert r.is_correct

def test_synonym_plasmodium():
    r = grade_diagnosis("plasmodium", "malaria")
    assert r.is_correct

def test_synonym_chikv():
    r = grade_diagnosis("chikv", "chikungunya")
    assert r.is_correct


# ---------------------------------------------------------------------------
# grade_diagnosis — partial and no match
# ---------------------------------------------------------------------------

def test_partial_match():
    r = grade_diagnosis("dengue hemorrhagic", "dengue")
    assert not r.is_correct
    assert 0.0 < r.score < 1.0

def test_no_match():
    r = grade_diagnosis("tuberculosis", "common cold")
    assert not r.is_correct
    assert r.score == MIN_SCORE

def test_empty_prediction():
    r = grade_diagnosis("", "dengue")
    assert not r.is_correct
    assert r.score == MIN_SCORE


# ---------------------------------------------------------------------------
# Per-task grader functions return float in (0, 1)
# ---------------------------------------------------------------------------

def test_all_per_task_graders_return_float():
    graders = [
        ("common cold", grade_easy_fever_cough),
        ("influenza", grade_medium_flu_vs_dengue),
        ("pneumonia", grade_medium_pneumonia),
        ("dengue", grade_hard_dengue_like),
        ("covid-19", grade_hard_covid_respiratory),
        ("malaria", grade_expert_malaria_mimic),
        ("typhoid", grade_expert_typhoid_enteric),
        ("chikungunya", grade_expert_chikungunya),
    ]
    for correct_label, fn in graders:
        score = fn(correct_label)
        assert isinstance(score, float), f"{fn.__name__} did not return float"
        assert 0.0 < score < 1.0, f"{fn.__name__} returned {score} outside (0,1)"

def test_all_per_task_graders_wrong_answer():
    graders = [
        grade_easy_fever_cough,
        grade_medium_flu_vs_dengue,
        grade_medium_pneumonia,
        grade_hard_dengue_like,
        grade_hard_covid_respiratory,
        grade_expert_malaria_mimic,
        grade_expert_typhoid_enteric,
        grade_expert_chikungunya,
    ]
    for fn in graders:
        score = fn("totally_wrong_answer_xyz")
        assert isinstance(score, float)
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# TASK_GRADERS dict
# ---------------------------------------------------------------------------

def test_task_graders_has_all_tasks():
    for name in list_task_names():
        assert name in TASK_GRADERS, f"Missing grader for task '{name}'"

def test_task_graders_return_float():
    for name, fn in TASK_GRADERS.items():
        case = CASE_BY_NAME[name]
        score = fn(case.target_diagnosis)
        assert isinstance(score, float)
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# Tasks / cases
# ---------------------------------------------------------------------------

def test_eight_tasks():
    assert len(CASES) == 8
    assert len(list_task_names()) == 8

def test_all_cases_have_key_questions():
    for case in CASES:
        assert len(case.key_questions) >= 2, f"{case.name} has too few key_questions"

def test_all_cases_have_hidden_facts():
    for case in CASES:
        assert len(case.hidden_facts) >= 5, f"{case.name} has too few hidden_facts"


# ---------------------------------------------------------------------------
# Trajectory grading
# ---------------------------------------------------------------------------

def test_trajectory_correct_all_keys():
    case = CASE_BY_NAME["easy_fever_cough"]
    score = grade_trajectory(case, case.key_questions, True, 3, 8)
    assert 0.0 < score < 1.0
    assert score > 0.7  # should be high

def test_trajectory_correct_no_keys():
    case = CASE_BY_NAME["easy_fever_cough"]
    score = grade_trajectory(case, [], True, 3, 8)
    assert 0.0 < score < 1.0

def test_trajectory_wrong_diagnosis():
    case = CASE_BY_NAME["easy_fever_cough"]
    score = grade_trajectory(case, case.key_questions, False, 3, 8)
    assert 0.0 < score < 1.0
    assert score < 0.5  # wrong diagnosis should be low


# ---------------------------------------------------------------------------
# Time decay
# ---------------------------------------------------------------------------

def test_time_decay_step1():
    assert time_decay_factor(1, 8) == 1.0

def test_time_decay_last_step():
    assert time_decay_factor(8, 8) == 0.5

def test_time_decay_middle():
    d = time_decay_factor(4, 8)
    assert 0.5 < d < 1.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
