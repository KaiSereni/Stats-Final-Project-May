import json
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

JSON_FILE_PATH = 'out.json'
GDP_THRESHOLD = 82769  # US GDP per capita
US_MEAN_AGE = 40
OREGON_STATE_KEY = "Oregon"
ALPHA_LEVEL = float(input("What should the alpha level be? (ex: 0.05)\n"))

if not (0 < ALPHA_LEVEL < 1):
    print("Invalid alpha level!")
    exit()


def load_data(filepath):
    """Loads state data from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file {filepath} is not a valid JSON.")
        return None


def perform_z_test_gdp_proportion(state_data):
    """
    Performs a one-sample z-test to check if the proportion of states
    with GDP per capita > GDP_THRESHOLD is 50%, with outputs following
    the State-Plan-Do-Conclude framework.
    """
    print("\n--- 1. One-Sample Z-Test for Proportion of States with High GDP ---")

    # --- STATE ---
    print("\nState:")
    print(f"- Research Question: Is the true proportion of U.S. states with GDP per capita greater than ${GDP_THRESHOLD:,} equal to 0.50?")
    print("- Population of Interest: All U.S. states.")
    print("- Variable: Indicator variable for each state, 1 if GDP per capita > threshold, 0 otherwise.")
    print("- Parameter of Interest: p = proportion of all U.S. states with GDP per capita > threshold.")

    if not state_data:
        print("  Cannot proceed: No data loaded.")
        return

    gdp_values = [details['gdp_per_capita']
                  for details in state_data.values() if 'gdp_per_capita' in details]
    if not gdp_values:
        print("  Cannot proceed: No 'gdp_per_capita' data found.")
        return

    n_total_states = len(gdp_values)
    n_high_gdp_states = sum(1 for gdp in gdp_values if gdp > GDP_THRESHOLD)
    hypothesized_proportion = 0.50

    # --- PLAN ---
    print("\nPlan:")
    print(f"- Statistical Method: One-sample z-test for a population proportion.")
    print(f"- Null Hypothesis (H₀): p = {hypothesized_proportion:.2f}")
    print(f"- Alternative Hypothesis (H₁): p ≠ {hypothesized_proportion:.2f} (two-sided).")
    print("- Significance level (α):", ALPHA_LEVEL)
    print("- Check Conditions:")
    print(f"    • Sample size: n = {n_total_states}")
    print(f"    • Expected successes: n * p₀ = {n_total_states} × {hypothesized_proportion:.2f} = {n_total_states * hypothesized_proportion:.1f}")
    print(f"    • Expected failures: n * (1 − p₀) = {n_total_states} × {1 - hypothesized_proportion:.2f} = {n_total_states * (1 - hypothesized_proportion):.1f}")
    print("    (Both should be ≥ 10 for normal approximation.)")

    if n_total_states == 0:
        print("  Cannot proceed: Zero states with GDP data.")
        return

    # --- DO ---
    print("\nDo:")
    print(f"- Number of states with GDP per capita > ${GDP_THRESHOLD:,}: {n_high_gdp_states}")
    print(f"- Total number of states with GDP data: {n_total_states}")
    sample_proportion = n_high_gdp_states / n_total_states
    print(f"- Sample proportion (p̂): {sample_proportion:.4f}")

    try:
        z_stat, p_value = proportions_ztest(
            count=n_high_gdp_states,
            nobs=n_total_states,
            value=hypothesized_proportion,
            alternative='two-sided'
        )
        print(f"- Test statistic (z): {z_stat:.4f}")
        print(f"- P-value: {p_value:.4f}")
    except Exception as e:
        print(f"  Error during z-test calculation: {e}")
        return

    # --- CONCLUDE ---
    print("\nConclude:")
    if p_value < ALPHA_LEVEL:
        print(f"- At α = {ALPHA_LEVEL}, p-value ({p_value:.4f}) < α. Reject H₀.")
        print(f"- There is significant evidence that the true proportion of states with GDP per capita > ${GDP_THRESHOLD:,} differs from {hypothesized_proportion:.2f}.")
    else:
        print(f"- At α = {ALPHA_LEVEL}, p-value ({p_value:.4f}) ≥ α. Fail to reject H₀.")
        print(f"- There is not significant evidence that the true proportion differs from {hypothesized_proportion:.2f}.")


def perform_one_sample_t_test_age(state_data):
    """
    Performs a one-sample t-test to compare state average ages against the US mean age,
    with outputs following the State-Plan-Do-Conclude framework.
    """
    print("\n--- 2. One-Sample T-Test for State Average Ages vs US Mean Age ---")

    # --- STATE ---
    print("\nState:")
    print(f"- Research Question: Is the true mean of state average ages different from the U.S. mean age of {US_MEAN_AGE:.2f} years?")
    print("- Population of Interest: All U.S. states.")
    print("- Variable: Average age in each state.")
    print(f"- Parameter of Interest: μ = true mean of state average ages (in years).")

    if not state_data:
        print("  Cannot proceed: No data loaded.")
        return

    ages = [details['avg_age']
            for details in state_data.values() if 'avg_age' in details]
    if not ages:
        print("  Cannot proceed: No 'avg_age' data found.")
        return

    n = len(ages)
    sample_mean = np.mean(ages)
    sample_sd = np.std(ages, ddof=1)

    # --- PLAN ---
    print("\nPlan:")
    print("- Statistical Method: One-sample t-test.")
    print(f"- Null Hypothesis (H₀): μ = {US_MEAN_AGE:.2f}")
    print(f"- Alternative Hypothesis (H₁): μ ≠ {US_MEAN_AGE:.2f} (two-sided).")
    print("- Significance level (α):", ALPHA_LEVEL)
    print("- Check Conditions:")
    print(f"    • Sample size: n = {n}")
    print("    • Data should be approximately normally distributed or n ≥ 30 (Central Limit Theorem).")
    print(f"    • Observations are independent (assuming random sampling of states).")

    # --- DO ---
    print("\nDo:")
    print(f"- U.S. Mean Age (μ₀): {US_MEAN_AGE:.2f} years")
    print(f"- Sample size (n): {n}")
    print(f"- Sample mean age: {sample_mean:.2f} years")
    print(f"- Sample standard deviation: {sample_sd:.2f} years")
    try:
        t_stat, p_value = stats.ttest_1samp(ages, US_MEAN_AGE)
        print(f"- Test statistic (t): {t_stat:.4f}")
        print(f"- P-value: {p_value:.4f}")
    except Exception as e:
        print(f"  Error during t-test calculation: {e}")
        return

    # --- CONCLUDE ---
    print("\nConclude:")
    if p_value < ALPHA_LEVEL:
        print(f"- At α = {ALPHA_LEVEL}, p-value ({p_value:.4f}) < α. Reject H₀.")
        print(f"- There is significant evidence that the true mean age of states differs from {US_MEAN_AGE:.2f} years.")
    else:
        print(f"- At α = {ALPHA_LEVEL}, p-value ({p_value:.4f}) ≥ α. Fail to reject H₀.")
        print(f"- There is not significant evidence that the true mean age differs from {US_MEAN_AGE:.2f} years.")


def perform_chi_squared_test_age_gdp(state_data):
    """
    Performs a Chi-squared test for independence between average age and GDP per capita categories,
    with outputs following the State-Plan-Do-Conclude framework.
    """
    print("\n--- 3. Chi-Squared Test for Independence: Age vs GDP per Capita Categories ---")

    # --- STATE ---
    print("\nState:")
    print("- Research Question: Are average age and GDP per capita category independent across states?")
    print("- Population of Interest: All U.S. states.")
    print("- Variables:")
    print("    • Age category: 'Low' if avg_age ≤ median, 'High' if avg_age > median.")
    print("    • GDP category: 'Low' if gdp_per_capita ≤ median, 'High' if gdp_per_capita > median.")
    print("- Parameter of Interest: Joint distribution of (Age category, GDP category).")

    if not state_data:
        print("  Cannot proceed: No data loaded.")
        return

    ages = []
    gdps = []
    for state_details in state_data.values():
        if 'avg_age' in state_details and 'gdp_per_capita' in state_details:
            ages.append(state_details['avg_age'])
            gdps.append(state_details['gdp_per_capita'])

    if len(ages) < 2 or len(gdps) < 2:
        print("  Cannot proceed: Insufficient data for both 'avg_age' and 'gdp_per_capita'.")
        return

    if len(ages) != len(gdps):
        print("  Warning: Mismatch in number of age and GDP data points. Truncating to common length.")
        min_len = min(len(ages), len(gdps))
        ages = ages[:min_len]
        gdps = gdps[:min_len]

    median_age = np.median(ages)
    median_gdp = np.median(gdps)

    # --- PLAN ---
    print("\nPlan:")
    print("- Statistical Method: Chi-squared test of independence on a 2×2 contingency table.")
    print("- Null Hypothesis (H₀): Age category and GDP category are independent.")
    print("- Alternative Hypothesis (H₁): Age category and GDP category are not independent.")
    print("- Significance level (α):", ALPHA_LEVEL)
    print("- Procedure to Create Contingency Table:")
    print(f"    • Age Low category: avg_age ≤ {median_age:.2f} → coded as 0; Age High: > {median_age:.2f} → coded as 1.")
    print(f"    • GDP Low category: gdp_per_capita ≤ ${median_gdp:,.2f} → coded as 0; GDP High: > ${median_gdp:,.2f} → coded as 1.")
    print("- Check Conditions:")
    print("    • Expected cell counts should be ≥ 5 for validity of chi-squared approximation.")

    # --- DO ---
    print("\nDo:")
    print(f"- Median Average Age used for categorization: {median_age:.2f} years")
    print(f"- Median GDP per Capita used for categorization: ${median_gdp:,.2f}")
    age_categories = [0 if age <= median_age else 1 for age in ages]
    gdp_categories = [0 if gdp <= median_gdp else 1 for gdp in gdps]

    contingency_table = np.zeros((2, 2), dtype=int)
    for i in range(len(ages)):
        contingency_table[age_categories[i], gdp_categories[i]] += 1

    print("\n  Contingency Table (rows = Age Low/High; columns = GDP Low/High):")
    print("             GDP Low    GDP High")
    print(f"Age Low      {contingency_table[0, 0]:<10} {contingency_table[0, 1]:<10}")
    print(f"Age High     {contingency_table[1, 0]:<10} {contingency_table[1, 1]:<10}")

    try:
        chi2_stat, p_value, dof, expected_freq = stats.chi2_contingency(contingency_table)
        print(f"\n- Test statistic (χ²): {chi2_stat:.4f}")
        print(f"- Degrees of Freedom: {dof}")
        print(f"- P-value: {p_value:.4f}")
        print("- Expected Frequencies Table:")
        print(expected_freq)
    except ValueError as e:
        print(f"  Error during Chi-squared test: {e}")
        print("  This might occur if any expected cell count is zero or data are too sparse.")
        return

    # --- CONCLUDE ---
    print("\nConclude:")
    if p_value < ALPHA_LEVEL:
        print(f"- At α = {ALPHA_LEVEL}, p-value ({p_value:.4f}) < α. Reject H₀.")
        print("- There is a statistically significant association between age category and GDP category.")
    else:
        print(f"- At α = {ALPHA_LEVEL}, p-value ({p_value:.4f}) ≥ α. Fail to reject H₀.")
        print("- There is no statistically significant association between age category and GDP category.")


def perform_two_sample_t_test_age_by_gdp(state_data):
    """
    Performs a two-sample t-test to compare average ages between high and low GDP states,
    with outputs following the State-Plan-Do-Conclude framework.
    """
    print("\n--- 4. Two-Sample T-Test: Average Age in High vs Low GDP States ---")

    # --- STATE ---
    print("\nState:")
    print(f"- Research Question: Do states with GDP per capita > ${GDP_THRESHOLD:,} have a different mean average age than states with GDP per capita ≤ ${GDP_THRESHOLD:,}?")
    print("- Population of Interest: All U.S. states.")
    print("- Variables: Avg_age in each state, grouped by GDP category (High vs Low).")
    print("- Parameter of Interest: μ_high − μ_low = difference in true mean ages between high-GDP and low-GDP state groups.")

    if not state_data:
        print("  Cannot proceed: No data loaded.")
        return

    high_gdp_ages = []
    low_gdp_ages = []

    for details in state_data.values():
        if 'avg_age' in details and 'gdp_per_capita' in details:
            if details['gdp_per_capita'] > GDP_THRESHOLD:
                high_gdp_ages.append(details['avg_age'])
            else:
                low_gdp_ages.append(details['avg_age'])

    if not high_gdp_ages or not low_gdp_ages:
        print("  Cannot proceed: Insufficient data in one or both GDP groups.")
        return

    n_high = len(high_gdp_ages)
    n_low = len(low_gdp_ages)
    mean_high = np.mean(high_gdp_ages)
    mean_low = np.mean(low_gdp_ages)
    sd_high = np.std(high_gdp_ages, ddof=1)
    sd_low = np.std(low_gdp_ages, ddof=1)

    # --- PLAN ---
    print("\nPlan:")
    print("- Statistical Method: Two-sample t-test (unequal variances).")
    print("- Null Hypothesis (H₀): μ_high = μ_low (no difference).")
    print("- Alternative Hypothesis (H₁): μ_high ≠ μ_low (two-sided).")
    print("- Significance level (α):", ALPHA_LEVEL)
    print("- Check Conditions:")
    print(f"    • Sample sizes: n_high = {n_high}, n_low = {n_low}")
    print("    • Observations within each group approximately normal or n ≥ 30.")
    print("    • Samples are independent.")
    print("    • Variances need not be equal (Welch’s t-test).")

    # --- DO ---
    print("\nDo:")
    print(f"- Number of high GDP states: {n_high}")
    print(f"- Number of low GDP states: {n_low}")
    print(f"- Mean age of high GDP states: {mean_high:.2f} years")
    print(f"- Mean age of low GDP states: {mean_low:.2f} years")
    print(f"- Sample SD of high GDP ages: {sd_high:.2f}")
    print(f"- Sample SD of low GDP ages: {sd_low:.2f}")
    try:
        t_stat, p_value = stats.ttest_ind(high_gdp_ages, low_gdp_ages, equal_var=False)
        print(f"- Test statistic (t): {t_stat:.4f}")
        print(f"- P-value: {p_value:.4f}")
    except Exception as e:
        print(f"  Error during two-sample t-test: {e}")
        return

    # --- CONCLUDE ---
    print("\nConclude:")
    if p_value < ALPHA_LEVEL:
        print(f"- At α = {ALPHA_LEVEL}, p-value ({p_value:.4f}) < α. Reject H₀.")
        print("- There is significant evidence that mean ages differ between high-GDP and low-GDP states.")
    else:
        print(f"- At α = {ALPHA_LEVEL}, p-value ({p_value:.4f}) ≥ α. Fail to reject H₀.")
        print("- There is not significant evidence that mean ages differ between high-GDP and low-GDP states.")


def main():
    """Main function to run the statistical analyses."""
    print("Starting Statistical Analysis Script...")
    state_data = load_data(JSON_FILE_PATH)

    if state_data:
        perform_z_test_gdp_proportion(state_data)
        perform_one_sample_t_test_age(state_data)
        perform_chi_squared_test_age_gdp(state_data)
        perform_two_sample_t_test_age_by_gdp(state_data)
    else:
        print("Script cannot proceed due to data loading issues.")

    print("\nStatistical Analysis Script Finished.")


if __name__ == "__main__":
    main()
