import os

if __name__ == "__main__":
    # Check if the requirements are installed
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()

    # Install each requirement using pip
    for req in requirements:
        os.system(f"pip install {req}")

    # Print a message indicating that all requirements are installed
    print("All requirements are installed")