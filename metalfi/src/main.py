import time

from metalfi.src.controller import Controller


class Main(object):

    @staticmethod
    def main():
        # Calculate meta-datasets (if necessary)
        start = time.time()
        c = Controller()
        end = time.time()
        print(end - start)

        # Train meta-models (if necessary)
        start = time.time()
        c.trainMetaModel()
        end = time.time()
        print(end - start)

        # Load trained meta-models from storage and get evaluation results
        start = time.time()
        c.evaluate(["Titanic", "Iris", "Cancer", "Wine", "Boston", "cloud", "backache", "primary-tumor",
                    "EgyptianSkulls", "SPECTF", "cpu", "bodyfat", "Engine1", "ESL", "ilpd-numeric", "credit-approval",
                    "vowel", "socmob", "ERA", "LEV", "credit-g", "cmc", "ozone-level-8hr", "wind", "mammography",
                    "bank8FM"])
        end = time.time()
        print(end - start)


if __name__ == '__main__':
    Main().main()
