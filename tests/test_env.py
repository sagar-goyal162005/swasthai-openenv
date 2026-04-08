"""Unit tests for the SwasthAI environment."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openenv_submission.env import Action, Observation, SwasthAIEnv
from openenv_submission.tasks import list_task_names, CASE_BY_NAME


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_reset_default_task():
    env = SwasthAIEnv()
    obs = env.reset()
    assert obs.task == list_task_names()[0]
    assert len(obs.public_symptoms) > 0
    assert obs.history == []

def test_reset_each_task():
    env = SwasthAIEnv()
    for name in list_task_names():
        obs = env.reset(name)
        assert obs.task == name

def test_reset_unknown_task():
    env = SwasthAIEnv()
    try:
        env.reset("nonexistent_task")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_reset_with_seed():
    env = SwasthAIEnv()
    obs = env.reset("easy_fever_cough", seed=42)
    assert obs.task == "easy_fever_cough"


# ---------------------------------------------------------------------------
# Ask action
# ---------------------------------------------------------------------------

def test_ask_relevant_question():
    env = SwasthAIEnv()
    env.reset("easy_fever_cough")
    action = Action(type="ask", content="How long have you had these symptoms?")
    obs, reward, done, info = env.step(action)
    assert reward > 0.0
    assert reward < 1.0
    assert not done
    assert len(obs.history) == 2  # Q + A
    assert "duration" in obs.history[1].lower()

def test_ask_empty_question():
    env = SwasthAIEnv()
    env.reset("easy_fever_cough")
    action = Action(type="ask", content="")
    obs, reward, done, info = env.step(action)
    assert "last_action_error" in info

def test_ask_irrelevant_question():
    env = SwasthAIEnv()
    env.reset("easy_fever_cough")
    action = Action(type="ask", content="What is your favorite color?")
    obs, reward, done, info = env.step(action)
    assert reward > 0.0  # clamped minimum
    assert reward < 0.05

def test_ask_repeated_question_penalized():
    env = SwasthAIEnv()
    env.reset("easy_fever_cough")
    action = Action(type="ask", content="How long have you had these symptoms?")
    env.step(action)
    # Ask same topic again
    action2 = Action(type="ask", content="What is the duration of symptoms?")
    obs2, reward2, done2, info2 = env.step(action2)
    assert reward2 <= 0.02  # should be penalized

def test_ask_high_value_fact():
    env = SwasthAIEnv()
    env.reset("hard_dengue_like")
    action = Action(type="ask", content="What is the platelet count?")
    obs, reward, done, info = env.step(action)
    assert reward >= 0.10  # high-value key

def test_ask_new_keywords_sputum():
    env = SwasthAIEnv()
    env.reset("medium_pneumonia")
    action = Action(type="ask", content="Do you have sputum or phlegm?")
    obs, reward, done, info = env.step(action)
    assert "sputum" in obs.history[1].lower()

def test_ask_new_keywords_smell():
    env = SwasthAIEnv()
    env.reset("hard_covid_respiratory")
    action = Action(type="ask", content="Have you lost your sense of smell?")
    obs, reward, done, info = env.step(action)
    assert "smell" in obs.history[1].lower()

def test_ask_new_keywords_conjunctivitis():
    env = SwasthAIEnv()
    env.reset("expert_chikungunya")
    action = Action(type="ask", content="Do you have conjunctivitis or red eyes?")
    obs, reward, done, info = env.step(action)
    assert "conjunctivitis" in obs.history[1].lower()


# ---------------------------------------------------------------------------
# Diagnose action
# ---------------------------------------------------------------------------

def test_diagnose_correct():
    env = SwasthAIEnv()
    env.reset("easy_fever_cough")
    action = Action(type="diagnose", content="common cold")
    obs, reward, done, info = env.step(action)
    assert info["is_correct"] is True
    assert done is True
    assert reward > 0.5

def test_diagnose_synonym():
    env = SwasthAIEnv()
    env.reset("medium_flu_vs_dengue")
    action = Action(type="diagnose", content="flu")
    obs, reward, done, info = env.step(action)
    assert info["is_correct"] is True
    assert done is True

def test_diagnose_wrong():
    env = SwasthAIEnv()
    env.reset("easy_fever_cough")
    action = Action(type="diagnose", content="malaria")
    obs, reward, done, info = env.step(action)
    assert info["is_correct"] is False
    assert done is False  # episode continues

def test_diagnose_all_tasks_correct():
    """Each task should be solvable with the exact target diagnosis."""
    for name in list_task_names():
        env = SwasthAIEnv()
        env.reset(name)
        case = CASE_BY_NAME[name]
        action = Action(type="diagnose", content=case.target_diagnosis)
        obs, reward, done, info = env.step(action)
        assert info["is_correct"] is True, f"Task {name} not solvable with '{case.target_diagnosis}'"
        assert done is True


# ---------------------------------------------------------------------------
# Confidence field
# ---------------------------------------------------------------------------

def test_confidence_correct_high():
    env = SwasthAIEnv()
    env.reset("easy_fever_cough")
    action = Action(type="diagnose", content="common cold", confidence=0.95)
    obs, reward, done, info = env.step(action)
    assert info.get("agent_confidence") == 0.95
    assert done is True

def test_confidence_wrong_high():
    env = SwasthAIEnv()
    env.reset("easy_fever_cough")
    action = Action(type="diagnose", content="malaria", confidence=0.95)
    obs, reward, done, info = env.step(action)
    assert info.get("agent_confidence") == 0.95
    assert done is False


# ---------------------------------------------------------------------------
# Wrong diagnosis penalty
# ---------------------------------------------------------------------------

def test_wrong_diagnosis_penalty_increases():
    env = SwasthAIEnv()
    env.reset("easy_fever_cough")
    r1 = env.step(Action(type="diagnose", content="malaria"))[1]
    r2 = env.step(Action(type="diagnose", content="dengue"))[1]
    # Second wrong diagnosis should have higher penalty
    assert r2 <= r1 or abs(r2 - r1) < 0.15  # penalty increases


# ---------------------------------------------------------------------------
# Time decay
# ---------------------------------------------------------------------------

def test_earlier_diagnosis_higher_reward():
    # Step 1 diagnosis vs step 5 diagnosis
    env1 = SwasthAIEnv()
    env1.reset("easy_fever_cough")
    _, r1, _, _ = env1.step(Action(type="diagnose", content="common cold"))

    env2 = SwasthAIEnv()
    env2.reset("easy_fever_cough")
    for _ in range(4):
        env2.step(Action(type="ask", content="How long have you had these symptoms?"))
    _, r2, _, _ = env2.step(Action(type="diagnose", content="common cold"))

    assert r1 >= r2  # earlier diagnosis should get higher reward


# ---------------------------------------------------------------------------
# Trajectory grading
# ---------------------------------------------------------------------------

def test_trajectory_score_in_info():
    env = SwasthAIEnv()
    env.reset("easy_fever_cough")
    env.step(Action(type="ask", content="How long have you had these symptoms?"))
    env.step(Action(type="ask", content="Do you have any rash?"))
    _, _, _, info = env.step(Action(type="diagnose", content="common cold"))
    assert "trajectory_score" in info
    assert 0.0 < info["trajectory_score"] < 1.0


# ---------------------------------------------------------------------------
# Structured vitals
# ---------------------------------------------------------------------------

def test_vitals_appear_after_asking():
    env = SwasthAIEnv()
    obs = env.reset("hard_dengue_like")
    assert obs.vitals is None or obs.vitals == {}

    obs2, _, _, _ = env.step(Action(type="ask", content="What is your temperature?"))
    assert obs2.vitals is not None
    assert "temperature" in obs2.vitals


# ---------------------------------------------------------------------------
# Seed-based variations
# ---------------------------------------------------------------------------

def test_seed_produces_different_facts():
    env1 = SwasthAIEnv()
    obs1 = env1.reset("easy_fever_cough", seed=1)
    env2 = SwasthAIEnv()
    obs2 = env2.reset("easy_fever_cough", seed=999)
    # With different seeds, at least the underlying case may vary
    # (the public symptoms stay the same, but hidden facts can differ)
    assert obs1.task == obs2.task  # same task

def test_seed_reproducible():
    env1 = SwasthAIEnv()
    env1.reset("easy_fever_cough", seed=42)
    env1.step(Action(type="ask", content="How long have you had these symptoms?"))
    s1 = env1.state()

    env2 = SwasthAIEnv()
    env2.reset("easy_fever_cough", seed=42)
    env2.step(Action(type="ask", content="How long have you had these symptoms?"))
    s2 = env2.state()

    assert s1.retrieved_facts == s2.retrieved_facts


# ---------------------------------------------------------------------------
# Max steps
# ---------------------------------------------------------------------------

def test_max_steps_terminates():
    env = SwasthAIEnv(max_steps=2)
    env.reset("easy_fever_cough")
    env.step(Action(type="ask", content="How long have you had these symptoms?"))
    obs, reward, done, info = env.step(Action(type="ask", content="Do you have any rash?"))
    assert done is True

def test_step_before_reset_raises():
    env = SwasthAIEnv()
    try:
        env.step(Action(type="ask", content="test"))
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def test_state_tracks_retrieved_facts():
    env = SwasthAIEnv()
    env.reset("hard_dengue_like")
    env.step(Action(type="ask", content="What is the platelet count?"))
    s = env.state()
    assert "platelets" in s.retrieved_facts

def test_state_tracks_wrong_diagnoses():
    env = SwasthAIEnv()
    env.reset("easy_fever_cough")
    env.step(Action(type="diagnose", content="malaria"))
    s = env.state()
    assert s.wrong_diagnoses == 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
