"""Model summary sidebar and model-performance tab."""

from pathlib import Path

import pandas as pd
import streamlit as st

from heart.predict import HeartModel


def sidebar(model: HeartModel, metadata: dict) -> None:
    with st.sidebar:
        st.header("Model")
        st.write(f"**{model.model_name.replace('_', ' ').title()}** · v{model.version}")
        cv = metadata["cv"]["results"][model.model_name]
        test = metadata["test_metrics"]
        st.metric("CV ROC-AUC", f"{cv['roc_auc']['mean']:.3f}", help=metadata["cv"]["scheme"])
        st.caption(f"± {cv['roc_auc']['std']:.3f} across folds")
        col1, col2 = st.columns(2)
        col1.metric("Test recall", f"{test['recall']:.2f}")
        col2.metric("Test precision", f"{test['precision']:.2f}")
        st.caption(f"Held-out test set: {test['n_samples']} patients.")
        st.metric(
            "Decision threshold",
            f"{model.threshold:.2f}",
            help=f"Tuned so out-of-fold recall reaches {metadata['target_recall']:.0%}",
        )
        st.caption(
            f"Features: {metadata['feature_set']} · Calibration: {metadata['calibration']} · "
            f"Optuna trials/family: {metadata['tuning']['n_trials']}"
        )


PLOTS = [
    ("model_comparison.png", "Model comparison"),
    ("roc_curve.png", "ROC curve"),
    ("pr_curve.png", "Precision-recall"),
    ("calibration.png", "Calibration"),
    ("threshold_tradeoff.png", "Threshold trade-off"),
    ("confusion_matrix.png", "Confusion matrix"),
    ("shap_summary.png", "Global SHAP importance"),
]


def performance_tab(metadata: dict, reports_dir: Path, model_card_path: Path) -> None:
    st.subheader("How the model was chosen")
    st.caption(
        f"{len(metadata['cv']['results'])} model families tuned with Optuna, then compared with "
        f"{metadata['cv']['scheme']} on the training set only."
    )
    rows = [
        {
            "model": name.replace("_", " "),
            "ROC-AUC": round(r["roc_auc"]["mean"], 3),
            "± std": round(r["roc_auc"]["std"], 3),
            "recall @0.5": round(r["recall"]["mean"], 3),
            "Brier": round(r["brier_score"]["mean"], 3),
        }
        for name, r in metadata["cv"]["results"].items()
    ]
    st.dataframe(pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False), hide_index=True)

    test, ci = metadata["test_metrics"], metadata["test_ci_95"]
    st.subheader(f"Held-out test set ({test['n_samples']} patients)")
    cols = st.columns(4)
    for col, key, label in zip(
        cols,
        ["roc_auc", "recall", "specificity", "precision"],
        ["ROC-AUC", "Recall", "Specificity", "Precision"],
        strict=True,
    ):
        col.metric(label, f"{test[key]:.2f}")
        col.caption(f"95% CI {ci[key][0]:.2f} to {ci[key][1]:.2f}")

    available = [(reports_dir / f, t) for f, t in PLOTS if (reports_dir / f).exists()]
    for i in range(0, len(available), 2):
        pair = st.columns(2)
        for col, (path, title) in zip(pair, available[i : i + 2], strict=False):
            col.markdown(f"**{title}**")
            col.image(str(path))

    st.subheader("Subgroups (out-of-fold, training set)")
    st.dataframe(
        pd.DataFrame(metadata["subgroups_oof"]).T.rename_axis("group").reset_index(),
        hide_index=True,
    )

    if model_card_path.exists():
        with st.expander("Full model card"):
            st.markdown(model_card_path.read_text(encoding="utf-8"))
