## 2026-02-17 - Dashboard Onboarding & Formatting
**Learning:** Adding a "Demo Mode" toggle is a critical UX pattern for data-heavy dashboards, allowing immediate exploration and onboarding without requiring a live data connection.
**Action:** When building complex dashboards, always include a `use_demo` toggle or fallback mock data to prevent "Empty State" fatigue during development and for new users.

**Learning:** Streamlit's `st.dataframe` works best with `Pandas Styler` for conditional coloring (e.g. green/red EV), while simple formatting can be chained with `.format()`. `column_config` is powerful but incompatible with Styler in some contexts.
**Action:** For tables requiring BOTH complex coloring and formatting, chain `.style.map(...).format(...)` rather than mixing Styler and `column_config`.
