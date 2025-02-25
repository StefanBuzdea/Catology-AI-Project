from Catology.Normalize_Data.constants import data_analysis_file
from Catology.Normalize_Data.constants import file_path, output_file_path
from Catology.data_import import import_data_from_excel
from Catology.Normalize_Data.data_conversions import converts_and_validations
from Catology.Normalize_Data.data_analysis import display_instances_per_class, display_distinct_values, display_distinct_values_per_class
from Catology.Normalize_Data.visualization import display_value_distributions, plot_distribution_with_pie, plot_heatmap_race_correlation
from Catology.data_export import save_to_text, save_data_to_excel

def normalize_data():
    excel_data = import_data_from_excel() # file_path
    data = converts_and_validations() # excel_data

    # Displays the number of instances for each class
    # instances_per_class = display_instances_per_class(data)

    # Displays the distinct values and their frequencies across the entire file
    # distinct_values = display_distinct_values(data)

    # Display distinct values and their frequencies, and calculate entropy for each breed
    # distinct_values_per_class, entropies_per_class = display_distinct_values_per_class(data)

    # Save results to a text file
    # save_to_text(data_analysis_file, instances_per_class, distinct_values, distinct_values_per_class,
    #              entropies_per_class)

    # print(f"Results saved to {data_analysis_file}")

    # print(data['Race'])

    # display_value_distributions(data)
    # plot_heatmap_race_correlation(data)
    # plot_distribution_with_pie(data)

    # data = convert_to_numeric(data)

    save_data_to_excel(data, output_file_path)