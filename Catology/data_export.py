
def save_data_to_excel(data, output_file_path):
    data.to_excel(output_file_path, index=False)


def save_to_text(file_name, instances_per_class, distinct_values, distinct_values_per_class, min_entropies_per_class):
    with open(file_name, 'w', encoding='utf-8') as f:

        f.write("Number of instances per cat breed (class):\n")
        f.write(instances_per_class.to_string())
        f.write("\n\n")

        f.write("Distinct values and their frequencies for each attribute (overall):\n")
        for column, value_counts in distinct_values.items():
            f.write(f"Attribute: {column}\n")
            f.write(value_counts.to_string(index=True, header=False))
            f.write("\n\n")

        f.write("Distinct values and their frequencies per breed (class):\n")
        for breed, attributes in distinct_values_per_class.items():
            f.write("\n\n")
            f.write("=================================================================================================\n")
            f.write(f"Breed: {breed}\n")
            f.write("=================================================================================================\n\n")

            for column, value_counts in attributes.items():
                f.write("\n\n")
                f.write(f"Attribute: {column}\n")
                f.write(value_counts.to_string(index=True, header=False))
                f.write("\n")

            f.write("\n")

        f.write("\n\n")
        f.write("Minimum entropy values for each breed (class):\n")
        for breed, min_entropy_info in min_entropies_per_class.items():
            f.write("\n\n")
            f.write("=================================================================================================\n")
            f.write(f"Breed: {breed}\n")
            f.write("=================================================================================================\n\n")

            f.write(f"Attribute with the lowest entropy: {min_entropy_info['attribute']}\n")
            f.write(f"Minimum Entropy: {min_entropy_info['entropy']:.4f}\n")
            f.write("\n")