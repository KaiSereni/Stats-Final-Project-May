import json
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

JSON_FILE_PATH = 'out.json'
GDP_THRESHOLD = 82769
US_MEAN_AGE = 39.6
OREGON_STATE_KEY = "Oregon"
ALPHA_LEVEL = 0.05

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
    with GDP per capita > GDP_THRESHOLD is 50%.
    """
    print("\n--- 1. One-Sample Z-Test for Proportion of States with High GDP ---")
    if not state_data:
        print("Cannot perform test: No data loaded.")
        return

    gdp_values = [details['gdp_per_capita'] for details in state_data.values() if 'gdp_per_capita' in details]
    if not gdp_values:
        print("Cannot perform test: No 'gdp_per_capita' data found in states.")
        return

    n_total_states = len(gdp_values)
    n_high_gdp_states = sum(1 for gdp in gdp_values if gdp > GDP_THRESHOLD)
    
    hypothesized_proportion = 0.50

    print(f"Hypothesized proportion (p0): {hypothesized_proportion:.2f}")
    print(f"Number of states with GDP per capita > ${GDP_THRESHOLD:,}: {n_high_gdp_states}")
    print(f"Total number of states with GDP data: {n_total_states}")

    if n_total_states == 0:
        print("Cannot perform test: Zero states with GDP data.")
        return

    sample_proportion = n_high_gdp_states / n_total_states
    print(f"Sample proportion (p_hat): {sample_proportion:.4f}")

    # Perform the z-test
    # H0: p = hypothesized_proportion
    # HA: p != hypothesized_proportion (two-sided)
    try:
        z_stat, p_value = proportions_ztest(count=n_high_gdp_states, 
                                            nobs=n_total_states, 
                                            value=hypothesized_proportion,
                                            alternative='two-sided')
        print(f"Z-statistic: {z_stat:.4f}")
        print(f"P-value: {p_value:.4f}")

        if p_value < ALPHA_LEVEL:
            print(f"Conclusion: At alpha={ALPHA_LEVEL}, we reject the null hypothesis.")
            print(f"There is significant evidence that the true proportion of states with GDP per capita > ${GDP_THRESHOLD:,} is different from {hypothesized_proportion:.2f}.")
        else:
            print(f"Conclusion: At alpha={ALPHA_LEVEL}, we fail to reject the null hypothesis.")
            print(f"There is not significant evidence that the true proportion of states with GDP per capita > ${GDP_THRESHOLD:,} differs from {hypothesized_proportion:.2f}.")
    except Exception as e:
        print(f"Error during z-test calculation: {e}")


def analyze_oregon_age(state_data):
    """
    Analyzes Oregon's average age in comparison to the US average age.
    Explains limitations of a direct one-sample t-test on a single value.
    """
    print(f"\n--- 2. Analysis of {OREGON_STATE_KEY}'s Average Age vs. US Average Age ({US_MEAN_AGE} years) ---")
    if not state_data:
        print("Cannot perform analysis: No data loaded.")
        return

    if OREGON_STATE_KEY not in state_data or 'avg_age' not in state_data[OREGON_STATE_KEY]:
        print(f"Cannot perform analysis: Data for '{OREGON_STATE_KEY}' or its 'avg_age' not found in the JSON.")
        return

    oregon_avg_age = state_data[OREGON_STATE_KEY]['avg_age']
    print(f"{OREGON_STATE_KEY}'s average age: {oregon_avg_age:.1f} years")
    print(f"US average age (population mean): {US_MEAN_AGE:.1f} years")

    if oregon_avg_age < US_MEAN_AGE:
        comparison = "lower than"
    elif oregon_avg_age > US_MEAN_AGE:
        comparison = "higher than"
    else:
        comparison = "equal to"
    print(f"Direct comparison: {OREGON_STATE_KEY}'s average age is {comparison} the US average age.")

    print("\nNote on the requested One-Sample T-Test:")
    print("A one-sample t-test is typically used to compare the mean of a *sample* of data (which has inherent variability and a standard deviation) against a known population mean.")
    print(f"The provided data for {OREGON_STATE_KEY} is a single 'avg_age' value ({oregon_avg_age:.1f}).")
    print("To perform a meaningful one-sample t-test for Oregon's age, we would ideally need a sample of individual age data points from Oregon to calculate a sample mean and sample standard deviation.")
    print("Applying a t-test directly to a single summary statistic (like a pre-calculated average without its own variance or sample size) is not standard practice, as the concept of sample variance required for the t-statistic is undefined for a single point.")
    print("Therefore, while we can compare the values directly, a formal t-test for significance on this single value against the population mean cannot be robustly performed without more detailed data from Oregon.")

def perform_chi_squared_test_age_gdp(state_data):
    """
    Performs a Chi-squared test for independence between average age and GDP per capita.
    Categories are determined by medians.
    """
    print("\n--- 3. Chi-Squared Test for Independence between Average Age and GDP per Capita ---")
    if not state_data:
        print("Cannot perform test: No data loaded.")
        return

    ages = []
    gdps = []
    for state_details in state_data.values():
        if 'avg_age' in state_details and 'gdp_per_capita' in state_details:
            ages.append(state_details['avg_age'])
            gdps.append(state_details['gdp_per_capita'])

    if len(ages) < 2 or len(gdps) < 2:
        print("Cannot perform test: Insufficient data for both 'avg_age' and 'gdp_per_capita' across states.")
        return
    
    if len(ages) != len(gdps):
        print("Warning: Mismatch in the number of age and GDP data points. Using the minimum common length.")
        min_len = min(len(ages), len(gdps))
        ages = ages[:min_len]
        gdps = gdps[:min_len]

    median_age = np.median(ages)
    median_gdp = np.median(gdps)

    print(f"Median Average Age: {median_age:.2f} years (used for categorization: Low <= median, High > median)")
    print(f"Median GDP per Capita: ${median_gdp:,.2f} (used for categorization: Low <= median, High > median)")

    age_categories = [0 if age <= median_age else 1 for age in ages]
    gdp_categories = [0 if gdp <= median_gdp else 1 for gdp in gdps]

    contingency_table = np.zeros((2, 2), dtype=int)
    for i in range(len(ages)):
        contingency_table[age_categories[i], gdp_categories[i]] += 1

    print("\nContingency Table (Age vs. GDP):")
    print("           GDP Low   GDP High")
    print(f"Age Low    {contingency_table[0,0]:<9} {contingency_table[0,1]:<9}")
    print(f"Age High   {contingency_table[1,0]:<9} {contingency_table[1,1]:<9}")

    # H0: Average age and GDP per capita are independent.
    # HA: Average age and GDP per capita are not independent.
    try:
        chi2_stat, p_value, dof, expected_freq = stats.chi2_contingency(contingency_table)
        print(f"\nChi-squared statistic: {chi2_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Degrees of freedom: {dof}")
        # print(f"Expected Frequencies Table:\n{expected_freq}")

        if p_value < ALPHA_LEVEL:
            print(f"Conclusion: At alpha={ALPHA_LEVEL}, we reject the null hypothesis.")
            print("There is a statistically significant relationship between average age and GDP per capita categories.")
        else:
            print(f"Conclusion: At alpha={ALPHA_LEVEL}, we fail to reject the null hypothesis.")
            print("There is no statistically significant relationship found between average age and GDP per capita categories.")
    except ValueError as e:
        print(f"Error during Chi-squared test: {e}")
        print("This might occur if the contingency table has sums of zero in rows/columns, often due to very small sample size or skewed data distribution after categorization.")


def main():
    """Main function to run the statistical analyses."""
    print("Starting Statistical Analysis Script...")
    state_data = load_data(JSON_FILE_PATH)

    if state_data:
        perform_z_test_gdp_proportion(state_data)
        analyze_oregon_age(state_data)
        perform_chi_squared_test_age_gdp(state_data)
    else:
        print("Script cannot proceed due to data loading issues.")
    
    print("\nStatistical Analysis Script Finished.")

if __name__ == "__main__":
    main()
