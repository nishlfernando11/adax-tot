def get_task(name):
    if name == 'game24':
        from src.tot.tasks.game24 import Game24Task
        return Game24Task()
    elif name == 'text':
        from src.tot.tasks.text import TextTask
        return TextTask()
    elif name == 'crosswords':
        from src.tot.tasks.crosswords import MiniCrosswordsTask
        return MiniCrosswordsTask()
    elif name == 'adax':
        from src.tot.tasks.adax import AdaXTask
        return AdaXTask()
    else:
        raise NotImplementedError