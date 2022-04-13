from setuptools import setup, find_packages


def main():
    scripts = []
    
    setup(
        name="LapStyle",
        version="1.0",
        author="Kirill, Rustem",
        package_dir={"": "src"},
        packages=find_packages("src"),
    )


if __name__ == "__main__":
    main()