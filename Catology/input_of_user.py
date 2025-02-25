from Catology.Language_Processing.generate_description_for_race import generate_description_for_class, \
    generate_comparison_between_classes
from Catology.Neuronal_Network.classify_an_instance import apply_nn_to_instance
from Catology.Normalize_Data.constants import conversion_map

reverse_race_map = {v: k for k, v in conversion_map['Race'].items() if v != -1}  # Exclude NSP


def display_breeds():
    """
    Display the list of breeds with their corresponding numeric codes.
    """
    print("Available breeds:")
    for number, breed in reverse_race_map.items():
        if breed == "Autre":
            breed = "Other"  # Înlocuim 'Autre' cu 'Other'
        print(f"{number}. {breed}")


def describe_breed(breed_number):
    """
    Generate and display a description for a specific breed based on its number.
    :param breed_number: Numeric code of the breed.
    """
    breed_name = reverse_race_map[breed_number]
    description = generate_description_for_class(breed_name)
    print(f"Description for {breed_name} ({breed_number}):")
    print(description)


def compare_breeds(breed_number1, breed_number2):
    """
    Generate and display a comparison between two breeds based on their numbers.
    :param breed_number1: Numeric code of the first breed.
    :param breed_number2: Numeric code of the second breed.
    """
    breed_name1 = reverse_race_map[breed_number1]
    breed_name2 = reverse_race_map[breed_number2]
    comparison = generate_comparison_between_classes(breed_name1, breed_name2)
    print(f"Comparison between {breed_name1} ({breed_number1}) and {breed_name2} ({breed_number2}):")
    print(comparison)


def input_of_user():
    """
    Main program for classifying cats.
    """
    print("Available functionalities:")
    print("---------------------------------------------------------------------------------")
    print("1. Write a description of a cat to find its breed")
    print("2. Select a breed to provide a description of that breed")
    print("3. Select 2 breeds to compare them")
    print("4. Exit program")
    print("---------------------------------------------------------------------------------")

    choice = input("Choose an option (type 1 for the first functionality, 2 for the second, 3 or 4 to exit): ")

    while 1:
        if choice == "1":
            apply_nn_to_instance()
        elif choice == "2":
            display_breeds()
            try:
                breed_number = int(input("Enter the number of the breed you want to describe: "))
                if breed_number in reverse_race_map:
                    describe_breed(breed_number)
                else:
                    print("Invalid breed number.")
            except ValueError:
                print("Please enter a valid number.")
        elif choice == "3":
            display_breeds()
            try:
                breed_numbers = input("Enter the numbers of two breeds you want to compare, separated by a space: ")
                breed_number1, breed_number2 = map(int, breed_numbers.split())
                if breed_number1 in reverse_race_map and breed_number2 in reverse_race_map:
                    compare_breeds(breed_number1, breed_number2)
                else:
                    print("Invalid breed numbers.")
            except ValueError:
                print("Please enter two valid numbers separated by a space.")
        elif choice == "4":
            break
        else:
            print("Invalid option. Exiting.")
            break

        print()
        choice = input("Choose an option (type 1 for the first functionality, 2 for the second, 3 or 4 to exit): ")