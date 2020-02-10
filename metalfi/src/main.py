from metalfi.src.controller import Controller


class Main(object):

    @staticmethod
    def main():
        c = Controller()
        c.trainMetaModel()
        models = c.loadModel(["Titanic", "Iris", "Cancer", "Wine", "Boston"])
        model, name = models[0]
        model.test()


if __name__ == '__main__':
    Main().main()
