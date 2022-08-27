import os


def get_dir(path):
    """ Make directory and return path
    :param path:
    :return:
    """

    if not os.path.isdir(path):
        os.mkdir(path)

    return path
