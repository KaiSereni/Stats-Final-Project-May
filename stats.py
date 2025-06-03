import json
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

JSON_FILE_PATH = 'out.json'
GDP_THRESHOLD = 82769 # US GDP per capita
US_MEAN_AGE = 40
OREGON_STATE_KEY = "Oregon"
ALPHA_LEVEL = float(input("What should the alpha level be? (ex: 0.05)\n"))

if not (ALPHA_LEVEL < 1 and ALPHA_LEVEL > 0):
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
    
def perform_one_sample_t_test_age(state_data):
    """
    Performs a one-sample t-test to compare state average ages against the US mean age.
    """
    print("\n--- 2. One-Sample T-Test for State Average Ages vs US Mean ---")
    if not state_data:
        print("Cannot perform test: No data loaded.")
        return

    ages = [details['avg_age'] for details in state_data.values() if 'avg_age' in details]
    if not ages:
        print("Cannot perform test: No 'avg_age' data found in states.")
        return

    print(f"US Mean Age (μ0): {US_MEAN_AGE:.2f} years")
    print(f"Number of states with age data: {len(ages)}")
    print(f"Sample mean age: {np.mean(ages):.2f} years")
    print(f"Sample standard deviation: {np.std(ages, ddof=1):.2f} years")

    # Perform the t-test
    # H0: μ = US_MEAN_AGE
    # HA: μ ≠ US_MEAN_AGE (two-sided)
    try:
        t_stat, p_value = stats.ttest_1samp(ages, US_MEAN_AGE)
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.4f}")

        if p_value < ALPHA_LEVEL:
            print(f"Conclusion: At alpha={ALPHA_LEVEL}, we reject the null hypothesis.")
            print(f"There is significant evidence that the true mean age of states differs from the US mean age of {US_MEAN_AGE:.2f} years.")
        else:
            print(f"Conclusion: At alpha={ALPHA_LEVEL}, we fail to reject the null hypothesis.")
            print(f"There is not significant evidence that the true mean age of states differs from the US mean age of {US_MEAN_AGE:.2f} years.")
    except Exception as e:
        print(f"Error during t-test calculation: {e}")

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

def perform_two_sample_t_test_age_by_gdp(state_data):
    """
    Performs a two-sample t-test to compare average ages between high and low GDP states.
    """
    print("\n--- 4. Two-Sample T-Test for Age Comparison between High and Low GDP States ---")
    if not state_data:
        print("Cannot perform test: No data loaded.")
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
        print("Cannot perform test: Insufficient data for both high and low GDP states.")
        return

    print(f"Number of high GDP states: {len(high_gdp_ages)}")
    print(f"Number of low GDP states: {len(low_gdp_ages)}")
    print(f"High GDP states mean age: {np.mean(high_gdp_ages):.2f} years")
    print(f"Low GDP states mean age: {np.mean(low_gdp_ages):.2f} years")

    # Perform the t-test
    # H0: μ_high = μ_low
    # HA: μ_high ≠ μ_low (two-sided)
    try:
        t_stat, p_value = stats.ttest_ind(high_gdp_ages, low_gdp_ages, equal_var=False)
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.4f}")

        if p_value < ALPHA_LEVEL:
            print(f"Conclusion: At alpha={ALPHA_LEVEL}, we reject the null hypothesis.")
            print("There is significant evidence that the mean age differs between high and low GDP states.")
        else:
            print(f"Conclusion: At alpha={ALPHA_LEVEL}, we fail to reject the null hypothesis.")
            print("There is not significant evidence that the mean age differs between high and low GDP states.")
    except Exception as e:
        print(f"Error during t-test calculation: {e}")

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
