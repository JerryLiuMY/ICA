import re


def activation2name(activation):
    """ Convert activation function to activation name
    :param activation: activation function for mlp
    :return:
    """

    activation_name = ''.join([_ for _ in re.sub("[\(\[].*?[\)\]]", "", str(activation)) if _.isalpha()])

    return activation_name
