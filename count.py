import csv
from collections import defaultdict

def calculate_average_tests_per_glass(file_path):
    # Dictionary to hold the count of tests for each glass type
    glass_test_counts = defaultdict(int)

    # Dictionary to hold the number of unique tests for each glass type
    glass_unique_tests = defaultdict(set)

    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            if len(row) > 4:  # Ensure the row has enough columns
                glass_info = row[4]  # The fifth column (index 4) contains the glass info

                # Split the glass info to separate the glass type and test type
                glass_parts = glass_info.split('_test')
                glass_type = glass_parts[0]

                # Increment the count for the glass type
                glass_test_counts[glass_type] += 1

                # Add the test number to the set of tests for the glass type
                if len(glass_parts) > 1:
                    test_number = glass_parts[1]
                    glass_unique_tests[glass_type].add(test_number)

    # Calculate the average number of tests per glass type
    average_tests_per_glass = {
        glass_type: len(tests) for glass_type, tests in glass_unique_tests.items()
    }

    return average_tests_per_glass

# Example usage
file_path = '/home/intra.cea.fr/ao280403/Bureau/ML Model/glass_data-13 - Glass Data(2).csv'  # Replace with the path to your CSV file
average_tests = calculate_average_tests_per_glass(file_path)
print(average_tests)
