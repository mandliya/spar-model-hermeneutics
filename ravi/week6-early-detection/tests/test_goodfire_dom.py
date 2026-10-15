import numpy as np

from early_detection.goodfire_dom import difference_of_means, fpr_threshold, lead_steps, summarize_leads


def test_difference_of_means_points_from_honest_to_hack():
    honest = np.zeros((4, 2))
    hack = np.ones((4, 2))
    direction = difference_of_means(hack, honest)
    assert np.allclose(direction, [1.0, 1.0])


def test_lead_is_the_step_gap_before_the_enacted_hack():
    passages = [
        {"step": 1, "passage_n": 1, "statuses": ["considered"], "score": 0.1},
        {"step": 3, "passage_n": 2, "statuses": ["considered"], "score": 2.0},
        {"step": 5, "passage_n": 3, "statuses": ["enacted"], "score": 3.0},
    ]
    lead = lead_steps(passages, threshold=1.0)
    assert lead["fired_early"]
    assert lead["lead_steps"] == 2
    assert summarize_leads([lead])["median_lead_steps"] == 2.0


def test_no_early_fire_is_a_result():
    passages = [
        {"step": 1, "passage_n": 1, "statuses": [], "score": 0.0},
        {"step": 4, "passage_n": 2, "statuses": ["attempted"], "score": 5.0},
    ]
    lead = lead_steps(passages, threshold=1.0)
    assert lead["fired_early"] is False
    assert lead["lead_steps"] is None


def test_each_method_keeps_its_own_threshold():
    from early_detection.goodfire_dom import score_report

    rows = []
    for index in range(20):
        rows.append(
            {
                "trace_id": "train",
                "consensus": "negative",
                "step": index,
                "passage_n": index,
                "statuses": [],
                "score": float(index),
            }
        )
    rows.append(
        {
            "trace_id": "hack",
            "consensus": "negative",
            "step": 0,
            "passage_n": 0,
            "statuses": [],
            "score": 0.0,
        }
    )
    rows.append(
        {
            "trace_id": "hack",
            "consensus": "positive",
            "step": 1,
            "passage_n": 1,
            "statuses": ["considered"],
            "score": 100.0,
        }
    )
    rows.append(
        {
            "trace_id": "hack",
            "consensus": "positive",
            "step": 4,
            "passage_n": 2,
            "statuses": ["enacted"],
            "score": 100.0,
        }
    )
    report = score_report(rows, {"train"}, {"hack"})
    assert report["test_auroc"] == 1.0
    assert report["lead"]["n_fired_before_action"] == 1
    assert report["lead"]["median_lead_steps"] == 3.0


def test_threshold_is_the_honest_tail():
    scores = np.arange(100, dtype=np.float64)
    cutoff = fpr_threshold(scores, fpr=0.05)
    assert cutoff >= 94
