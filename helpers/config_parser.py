from configparser import ConfigParser


def get_config_dict(filename, section):
    parser = ConfigParser()
    parser.read(filename)
    config_dictionary = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            config_dictionary[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
    return config_dictionary

def get_config_value(filename, section, attribute):
    parser = ConfigParser()
    parser.read(filename)
    config_dictionary = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            config_dictionary[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
    if(config_dictionary[attribute] is None):
        raise Exception('Attribute {0} not found in section {1} the {2} file'.format(attribute, section, filename))
    return config_dictionary[attribute]