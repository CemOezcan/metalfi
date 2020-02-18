from metalfi.src.controller import Controller


class Main(object):

    @staticmethod
    def main():
        c = Controller()

        # Calculate meta-datasets and train meta-models (if necessary)
        c.trainMetaModel()

        # Load trained meta-models from storage and get evaluation results
        c.evaluate(["Titanic", "Iris", "Cancer", "Wine", "Boston"])


if __name__ == '__main__':
    Main().main()
