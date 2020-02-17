from metalfi.src.controller import Controller


class Main(object):

    @staticmethod
    def main():
        c = Controller()

        # Calculate meta-datasets and train meta-models (if necessary)
        c.trainMetaModel()

        # Load trained meta-models from storage
        models = c.loadModel(["Titanic", "Iris", "Cancer", "Wine", "Boston"])

        # Test performance of the meta-model
        for model, name in models:
            model.test()


if __name__ == '__main__':
    Main().main()
