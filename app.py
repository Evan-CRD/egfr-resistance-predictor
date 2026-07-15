# Replace ONLY the "Prediction" page block in your app.py with the code below.

if page == "Prediction":
    st.header("Predict a mutation's relative drug-response profile")

    # Build and validate the published mutation-to-group mapping.
    mutation_group_counts = (
        data.groupby("mutation")["structure_group"]
        .nunique()
    )

    ambiguous_mutations = mutation_group_counts[
        mutation_group_counts > 1
    ].index.tolist()

    if ambiguous_mutations:
        st.error(
            "Some mutations have more than one structure-function group in "
            "the source data: "
            + ", ".join(ambiguous_mutations)
        )
        st.stop()

    mutation_to_group = (
        data[["mutation", "structure_group"]]
        .drop_duplicates()
        .set_index("mutation")["structure_group"]
        .to_dict()
    )

    mutation = st.selectbox(
        "EGFR mutation",
        sorted(mutation_to_group.keys()),
    )

    # Automatically assign the published group.
    published_group = mutation_to_group[mutation]

    st.markdown(
        f"""
        <div class="result">
            <strong>Published structure–function group</strong><br>
            <span style="font-size:1.25rem;">{published_group}</span><br><br>
            <span style="color:#aeb9c4;">
                This group is assigned from the published dataset and cannot
                be changed independently of the selected mutation.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "Predict response across all 18 TKIs",
        type="primary",
        use_container_width=True,
    ):
        if published_group not in metadata["known_structure_groups"]:
            st.error(
                f"The group '{published_group}' is not recognized by the trained model."
            )
            st.stop()

        frame = pd.DataFrame(
            {
                "drug": metadata["known_drugs"],
                "structure_group": published_group,
            }
        )

        prediction_features = frame[metadata["feature_columns"]]
        frame["predicted_log2_ratio"] = model.predict(prediction_features)

        frame["predicted_IC50_fold_vs_WT"] = (
            2 ** frame["predicted_log2_ratio"]
        )

        frame["interpretation"] = np.select(
            [
                frame["predicted_log2_ratio"] > 0,
                frame["predicted_log2_ratio"] < 0,
            ],
            [
                "More resistant than WT",
                "More sensitive than WT",
            ],
            default="Similar response to WT",
        )

        frame = frame.sort_values(
            "predicted_log2_ratio",
            ascending=False,
        )

        st.markdown(
            """
            <div class="result">
                <strong>Interpretation:</strong>
                Positive values mean that the mutant is predicted to require
                more drug than WT to reach the same IC50. Negative values
                indicate greater relative sensitivity.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.bar_chart(
            frame.set_index("drug")[["predicted_log2_ratio"]]
        )

        st.dataframe(
            frame,
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Download predicted profile",
            frame.to_csv(index=False),
            f"{mutation}_predicted_TKI_profile.csv",
            "text/csv",
        )
