from Catology.Language_Processing.nlp_main import nlp_main
from Catology.Neuronal_Network.classify_an_instance import apply_nn_to_instance
from Catology.Neuronal_Network.classify_an_instance_test import neuronal_network_instance_classify_test
from Catology.Neuronal_Network.optimize_parameters import optimize_hyperparameters
from Catology.Normalize_Data.normalize_data_main import normalize_data
from Catology.Neuronal_Network.neuronal_network_main import neuronal_network
from Catology.Language_Processing.generate_description_for_race import extract_proportions_with_frequent_values, \
    generate_description_for_class, generate_comparison_between_classes

# Main
from Catology.input_of_user import input_of_user

if __name__ == "__main__":

    # normalize_data()
    # optimize_hyperparameters()
    # neuronal_network()
    # nlp_main()
    # apply_nn_to_instance()
    # print(generate_description_for_class('EUR'))
    # print(generate_comparison_between_classes('EUR', 'NR'))
    input_of_user()

